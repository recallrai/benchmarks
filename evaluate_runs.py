"""
Evaluate LongMemEval runs using LLM-as-Judge with AWS Bedrock.

This script evaluates memory system runs using an LLM judge to assess retrieval quality.
Supports: mem0, nebula, recallr, supermemory, zep

Based on mem0 paper's LLM-as-judge evaluation methodology.

Usage:
    python evaluate_runs.py --provider recallr --benchmark-version oracle --requests-per-minute 50
"""

import os
import json
import time
import re
import argparse
import asyncio
from collections import defaultdict
import boto3
from pathlib import Path
from typing import Dict, List, Any, Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

# Bedrock Configuration (hardcoded)
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"
THINKING_BUDGET = 5_000

# Dummy bearer token to bypass boto3 validation
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "ABSKQmVkcm9ja0FQSUtleS05MGt5LWF0LTc2ODEyNTY0MTkyODplSDVaSWxpcHQ2bGNkQldJK0FvZ0hubFp4SHZXMkJrU3d1ZWhPM0tFYVpBS3c4cEJpb1lndEhoRWRidz0="

# LLM-as-Judge Prompt (adapted from mem0 paper)
JUDGE_PROMPT = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) retrieved memories/context that should help answer the question, and (4) the date when the question was asked.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:

Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The retrieved context might be much longer and include additional information. You should be generous with your grading - as long as the context contains information that touches on the same topic as the gold answer and would allow someone to correctly answer the question, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The retrieved context might include relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as the context refers to the same date or time period as the gold answer and would enable answering the question correctly, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

IMPORTANT: When the question involves relative time references (like "last Tuesday", "next month", "yesterday", etc.), use the question_date as the reference point to interpret what absolute date these relative references mean. For example, if the question is "When did I go to Hawaii last year?" and the question_date is "2024-05-15", then "last year" refers to 2023.

Now it's time for the real question:
Question: {question}
Question date: {question_date}
Gold answer: {gold_answer}
Retrieved context: {retrieved_context}

Provide your response in JSON format wrapped in ```json and ``` markdown tags with the following structure:
{{
    "reasoning": "A short (one sentence) explanation of your reasoning",
    "label": "CORRECT or WRONG"
}}

Do NOT include both CORRECT and WRONG in your label field, or it will break the evaluation. Choose only one."""


# Pydantic Models for Run Results (universal format)
class RetrievalResultItem(BaseModel):
    """Single retrieval strategy result"""
    strategy: str
    context: str
    latency: float  # in milliseconds

class RunResult(BaseModel):
    """Universal run result file structure"""
    index: int
    question_type: str
    question: str
    question_date: str
    ground_truth_answer: str
    metadata: Dict[str, Any] = {}
    retrieval_results: List[RetrievalResultItem]

class LLMJudgeResponse(BaseModel):
    """Response from LLM (without tag)"""
    reasoning: str
    label: Literal["CORRECT", "WRONG"]

class JudgeResponse(BaseModel):
    """Response from LLM judge"""
    tag: str
    label: Literal["CORRECT", "WRONG"]
    reasoning: str

class EvaluationResult(BaseModel):
    """Result from LLM judge evaluation"""
    index: int
    question_type: str
    question: str
    question_date: str
    ground_truth_answer: str
    judge_response: List[JudgeResponse]


class BedrockJudge:
    """LLM judge using AWS Bedrock reasoning models"""
    
    def __init__(self, api_key: str, region: str, model: str, thinking_budget: int):
        self.api_key = api_key
        self.region = region
        self.model = model
        self.thinking_budget = thinking_budget
        
        # Initialize Bedrock client
        self.client = boto3.client("bedrock-runtime", region_name=region)
        
        # Register authentication header injection
        event_system = self.client.meta.events
        event_system.register("before-send.bedrock-runtime.Converse", self._add_auth_header)
        
        # Rate limiting
        self._rate_limiter_lock = asyncio.Lock()
        self._last_request_time = 0
        self._min_interval = 0
    
    def set_rate_limit(self, requests_per_minute: int):
        """Set the rate limit for API requests."""
        self._min_interval = 60.0 / requests_per_minute
    
    def _add_auth_header(self, request, **kwargs):
        """Add Authorization header with API key"""
        request.headers['Authorization'] = f"Bearer {self.api_key}"
    
    async def _apply_rate_limit(self):
        """Apply rate limiting using async lock."""
        async with self._rate_limiter_lock:
            now = time.time()
            time_since_last = now - self._last_request_time
            if time_since_last < self._min_interval:
                await asyncio.sleep(self._min_interval - time_since_last)
            self._last_request_time = time.time()
    
    async def evaluate(self, question: str, question_date: str, gold_answer: str, retrieved_context: str) -> tuple[str, str]:
        """
        Evaluate retrieved context using LLM judge.
        
        Args:
            question: The question being asked
            question_date: The date when the question was asked
            gold_answer: Ground truth answer
            retrieved_context: Retrieved memories/context
            
        Returns:
            Tuple of (label, reasoning)
        """
        # Apply rate limiting before making request
        await self._apply_rate_limit()
        
        # Format the prompt
        prompt = JUDGE_PROMPT.format(
            question=question,
            question_date=question_date,
            gold_answer=gold_answer,
            retrieved_context=retrieved_context
        )
        
        # Build Bedrock request
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        kwargs = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {"temperature": 1.0},
            "additionalModelRequestFields": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget
                }
            }
        }
        
        # Retry logic for valid JSON
        max_retries = 3
        for _ in range(max_retries):
            # Run sync boto3 call in executor to not block event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.converse(**kwargs))
            
            # Extract response (reasoning block + text block)
            response_content = response["output"]["message"]["content"]
            text_content = response_content[1]["text"]
            
            # Try to extract JSON from markdown code blocks
            json_matches = re.findall(r'```json\s*(.*?)\s*```', text_content, re.DOTALL)
            
            if not json_matches:
                # Try parsing entire content as JSON
                try:
                    json.loads(text_content)
                    json_matches = [text_content]
                except json.JSONDecodeError:
                    pass
            
            if not json_matches:
                # Retry with correction prompt
                messages.append({"role": "assistant", "content": response_content})
                messages.append({
                    "role": "user",
                    "content": [{"text": "The response was not valid JSON wrapped in ```json tags. Please correct it."}]
                })
                kwargs["messages"] = messages
                continue
            
            # Parse and validate JSON with Pydantic
            json_content = json_matches[-1]
            try:
                response = LLMJudgeResponse.model_validate_json(json_content)
                return response.label, response.reasoning
                
            except Exception as e:
                # Retry with correction prompt
                messages.append({"role": "assistant", "content": response_content})
                messages.append({
                    "role": "user",
                    "content": [{"text": f"The JSON was invalid: {str(e)}. Please correct it."}]
                })
                kwargs["messages"] = messages
                continue
        
        raise ValueError(f"Failed to get valid response after {max_retries} attempts")


async def evaluate_single_file(
    result_file: Path,
    provider: str,
    provider_output_dir: Path,
    judge: BedrockJudge,
    log_lock: asyncio.Lock,
    stats: Dict[str, int]
) -> None:
    """
    Evaluate a single result file asynchronously.
    
    Args:
        result_file: Path to the result file
        provider: Provider name
        provider_output_dir: Output directory for evaluations
        judge: Bedrock judge instance
        log_lock: Lock for thread-safe logging
        stats: Shared statistics dictionary
    """
    filename = result_file.stem  # e.g. "5_single_hop_question_pass_2"
    
    # Parse filename: {index}_{question_type}_pass_{N}
    pass_match = re.match(r'^(\d+)_(.+)_pass_(\d+)$', filename)
    if not pass_match:
        # Also try legacy format: {index}_{question_type}
        legacy_match = re.match(r'^(\d+)_(.+)$', filename)
        if not legacy_match:
            async with log_lock:
                print(f"Invalid filename format: {filename}")
            stats["failed"] += 1
            return
        index = int(legacy_match.group(1))
        question_type = legacy_match.group(2)
        pass_idx = 1
    else:
        index = int(pass_match.group(1))
        question_type = pass_match.group(2)
        pass_idx = int(pass_match.group(3))
    
    # Load result
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
    except Exception as e:
        async with log_lock:
            print(f"Error loading {result_file}: {e}")
        stats["failed_results"] += 1
        return
    
    # Extract fields
    question = result_data.get("question")
    question_date = result_data.get("question_date", "Unknown")
    ground_truth = result_data.get("ground_truth_answer")
    
    if not question or not ground_truth:
        async with log_lock:
            print(f"Missing required fields in {result_file}")
        stats["failed_results"] += 1
        return
    
    stats["successful_results"] += 1
    
    # Build strategies from new array format (provider-agnostic)
    retrieval = result_data.get("retrieval_results", [])
    strategies = []
    for item in retrieval:
        name = item.get("strategy", "default")
        tag = name
        strategies.append((name, tag))
    
    # Collect latencies for each strategy
    for item in retrieval:
        name = item.get("strategy", "default")
        tag = name
        latency_ms = item.get("latency")
        if latency_ms is not None:
            stats["latencies"].setdefault(tag, []).append(latency_ms / 1000)
    
    # Determine eval path (one file per result, not per strategy)
    eval_path = provider_output_dir / f"{index}_{question_type}_pass_{pass_idx}.json"

    # Load existing judge responses to support partial re-evaluation
    existing_judge_responses: List[Dict[str, Any]] = []
    if eval_path.exists():
        try:
            with open(eval_path, 'r') as f:
                existing_eval = json.load(f)
            existing_judge_responses = existing_eval.get("judge_response", [])
        except Exception:
            existing_judge_responses = []

    existing_tags = {r.get("tag") for r in existing_judge_responses}

    # Start with existing responses and count them toward stats
    judge_responses = [JudgeResponse(**r) for r in existing_judge_responses]
    for r in existing_judge_responses:
        tag = r.get("tag", "default")
        label = r.get("label")
        if label == "CORRECT":
            stats["correct"] += 1
            stats["by_type"].setdefault(question_type, {"correct": 0, "wrong": 0, "total": 0})
            stats["by_type"][question_type]["correct"] += 1
            stats["by_type"][question_type]["total"] += 1
            stats["by_strategy"].setdefault(tag, {"correct": 0, "wrong": 0, "total": 0})
            stats["by_strategy"][tag]["correct"] += 1
            stats["by_strategy"][tag]["total"] += 1
            stats["by_type_and_strategy"].setdefault(question_type, {})
            stats["by_type_and_strategy"][question_type].setdefault(tag, {"correct": 0, "wrong": 0, "total": 0})
            stats["by_type_and_strategy"][question_type][tag]["correct"] += 1
            stats["by_type_and_strategy"][question_type][tag]["total"] += 1
        elif label == "WRONG":
            stats["wrong"] += 1
            stats["by_type"].setdefault(question_type, {"correct": 0, "wrong": 0, "total": 0})
            stats["by_type"][question_type]["wrong"] += 1
            stats["by_type"][question_type]["total"] += 1
            stats["by_strategy"].setdefault(tag, {"correct": 0, "wrong": 0, "total": 0})
            stats["by_strategy"][tag]["wrong"] += 1
            stats["by_strategy"][tag]["total"] += 1
            stats["by_type_and_strategy"].setdefault(question_type, {})
            stats["by_type_and_strategy"][question_type].setdefault(tag, {"correct": 0, "wrong": 0, "total": 0})
            stats["by_type_and_strategy"][question_type][tag]["wrong"] += 1
            stats["by_type_and_strategy"][question_type][tag]["total"] += 1

    missing_strategies = [(k, t) for k, t in strategies if t not in existing_tags]

    async with log_lock:
        if existing_judge_responses:
            print(f"[{filename}] Evaluating {len(missing_strategies)} missing strategy(ies) ({len(existing_judge_responses)} already done)...")
        else:
            print(f"[{filename}] Evaluating {len(strategies)} strategy(ies)...")

    # Evaluate each missing strategy
    for strategy_key, tag in missing_strategies:
        # Extract context from array
        try:
            item = next((r for r in retrieval if r.get("strategy") == strategy_key), None)
            context = item.get("context", "") if item else ""
        except Exception as e:
            async with log_lock:
                print(f"[{filename}] Error extracting context: {e}")
            stats["judge_failed"] += 1
            return
        
        # Evaluate with LLM judge (rate limited internally)
        try:
            label, reasoning = await judge.evaluate(question, question_date, ground_truth, context)
        except Exception as e:
            async with log_lock:
                print(f"[{filename}] Evaluation failed: {e}")
            stats["judge_failed"] += 1
            return
        
        # Add judge response to list
        judge_responses.append(JudgeResponse(tag=tag, label=label, reasoning=reasoning))
        
        # Update label counts (overall, per-type, per-strategy, and type×strategy)
        if label == "CORRECT":
            stats["correct"] += 1
            if question_type not in stats["by_type"]:
                stats["by_type"][question_type] = {"correct": 0, "wrong": 0, "total": 0}
            stats["by_type"][question_type]["correct"] += 1
            stats["by_type"][question_type]["total"] += 1
            if tag not in stats["by_strategy"]:
                stats["by_strategy"][tag] = {"correct": 0, "wrong": 0, "total": 0}
            stats["by_strategy"][tag]["correct"] += 1
            stats["by_strategy"][tag]["total"] += 1
            # Nested type×strategy
            if question_type not in stats["by_type_and_strategy"]:
                stats["by_type_and_strategy"][question_type] = {}
            if tag not in stats["by_type_and_strategy"][question_type]:
                stats["by_type_and_strategy"][question_type][tag] = {"correct": 0, "wrong": 0, "total": 0}
            stats["by_type_and_strategy"][question_type][tag]["correct"] += 1
            stats["by_type_and_strategy"][question_type][tag]["total"] += 1
        elif label == "WRONG":
            stats["wrong"] += 1
            if question_type not in stats["by_type"]:
                stats["by_type"][question_type] = {"correct": 0, "wrong": 0, "total": 0}
            stats["by_type"][question_type]["wrong"] += 1
            stats["by_type"][question_type]["total"] += 1
            if tag not in stats["by_strategy"]:
                stats["by_strategy"][tag] = {"correct": 0, "wrong": 0, "total": 0}
            stats["by_strategy"][tag]["wrong"] += 1
            stats["by_strategy"][tag]["total"] += 1
            # Nested type×strategy
            if question_type not in stats["by_type_and_strategy"]:
                stats["by_type_and_strategy"][question_type] = {}
            if tag not in stats["by_type_and_strategy"][question_type]:
                stats["by_type_and_strategy"][question_type][tag] = {"correct": 0, "wrong": 0, "total": 0}
            stats["by_type_and_strategy"][question_type][tag]["wrong"] += 1
            stats["by_type_and_strategy"][question_type][tag]["total"] += 1
    
    # Save evaluation result
    try:
        # Sort judge_responses in canonical tag order
        canonical_tags = [
            "low_latency", "balanced", "agentic",  # recallr
            "non_graph", "graph",  # mem0
            "low", "medium", "high",  # nebula
            "default",  # supermemory, zep
        ]
        tag_order = {tag: i for i, tag in enumerate(canonical_tags)}
        judge_responses.sort(key=lambda r: tag_order.get(r.tag, len(canonical_tags)))

        evaluation = EvaluationResult(
            index=index,
            question_type=question_type,
            question=question,
            question_date=question_date,
            ground_truth_answer=ground_truth,
            judge_response=judge_responses
        )
        
        # Save evaluation
        with open(eval_path, 'w') as f:
            json.dump(evaluation.model_dump(), f, indent=4)
        
        stats["evaluated"] += 1
        
        async with log_lock:
            print(f"[{filename}] ✓ Completed")
        
    except Exception as e:
        async with log_lock:
            print(f"[{filename}] Failed to create evaluation result: {e}")
        stats["judge_failed"] += 1


async def run_evaluation(
    provider: str,
    benchmark_version: str,
    runs_dir: Path,
    output_dir: Path,
    judge: BedrockJudge,
    skip_existing: bool
) -> Dict[str, Any]:
    """
    Evaluate all result files for a provider/benchmark asynchronously.
    
    Args:
        provider: Provider name (mem0, recallr, supermemory, zep)
        benchmark_version: Benchmark version (oracle, medium, small)
        runs_dir: Base directory for runs
        output_dir: Base directory for evaluations
        judge: Bedrock judge instance
        skip_existing: Skip files that already have evaluations
        
    Returns:
        Summary statistics
    """
    # Set up paths
    provider_runs_dir = runs_dir / provider / "longmemeval" / benchmark_version
    provider_output_dir = output_dir / provider / "longmemeval" / benchmark_version
    provider_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all result files (supports _pass_N naming)
    result_files = sorted(provider_runs_dir.glob("*.json"))
    
    if not result_files:
        print(f"No result files found in {provider_runs_dir}")
        return {}
    
    print(f"Found {len(result_files)} result files for {provider}/{benchmark_version}")
    
    # Track statistics (thread-safe with locks in evaluate_single_file)
    stats = {
        "total": len(result_files),
        "successful_results": 0,  # Files with valid retrieval results
        "failed_results": 0,  # Files that errored during benchmark run
        "evaluated": 0,  # Files we ran judge evaluation on
        "skipped": 0,  # Files we skipped (already evaluated)
        "judge_failed": 0,  # Files where judge evaluation failed
        "correct": 0,
        "wrong": 0,
        "by_type": {},  # Per-question-type breakdown: {type: {"correct": n, "wrong": n, "total": n}}
        "by_strategy": {},  # Per-strategy breakdown: {strategy: {"correct": n, "wrong": n, "total": n}}
        "by_type_and_strategy": {},  # Nested: {type: {strategy: {"correct": n, "wrong": n, "total": n}}}
        "latencies": {}  # Latency data per strategy: {strategy: [latency1, latency2, ...]}
    }
    
    # Create async lock for logging
    log_lock = asyncio.Lock()
    
    # Filter files to process
    files_to_process = []
    for result_file in result_files:
        filename = result_file.stem
        # Parse _pass_N naming
        pass_match = re.match(r'^(\d+)_(.+)_pass_(\d+)$', filename)
        if pass_match:
            index = int(pass_match.group(1))
            question_type = pass_match.group(2)
            pass_idx = int(pass_match.group(3))
        else:
            # Legacy format
            legacy_match = re.match(r'^(\d+)_(.+)$', filename)
            if not legacy_match:
                continue
            index = int(legacy_match.group(1))
            question_type = legacy_match.group(2)
            pass_idx = 1
        
        eval_path = provider_output_dir / f"{index}_{question_type}_pass_{pass_idx}.json"
        if skip_existing and eval_path.exists():
            # Check whether all expected strategies for this result are already evaluated
            fully_evaluated = False
            try:
                with open(result_file, 'r') as rf:
                    run_data = json.load(rf)
                with open(eval_path, 'r') as ef:
                    eval_data = json.load(ef)
                retrieval = run_data.get("retrieval_results", []) if "error" not in run_data else []
                expected_tags = {
                    f"{item.get('strategy', 'default')}_generated_context"
                    for item in retrieval
                }
                existing_tags = {r.get("tag") for r in eval_data.get("judge_response", [])}
                fully_evaluated = expected_tags.issubset(existing_tags)
            except Exception:
                pass

            if fully_evaluated:
                stats["skipped"] += 1

                # Count this file in successful results
                stats["successful_results"] += 1

                # Load existing evaluation and count results
                try:
                    with open(eval_path, 'r') as f:
                        eval_data = json.load(f)
                        judge_responses = eval_data.get("judge_response", [])
                        for response in judge_responses:
                            label = response.get("label")
                            tag = response.get("tag", "generated_context")
                            if label == "CORRECT":
                                stats["correct"] += 1
                                if question_type not in stats["by_type"]:
                                    stats["by_type"][question_type] = {"correct": 0, "wrong": 0, "total": 0}
                                stats["by_type"][question_type]["correct"] += 1
                                stats["by_type"][question_type]["total"] += 1
                                if tag not in stats["by_strategy"]:
                                    stats["by_strategy"][tag] = {"correct": 0, "wrong": 0, "total": 0}
                                stats["by_strategy"][tag]["correct"] += 1
                                stats["by_strategy"][tag]["total"] += 1
                                # Nested type×strategy
                                if question_type not in stats["by_type_and_strategy"]:
                                    stats["by_type_and_strategy"][question_type] = {}
                                if tag not in stats["by_type_and_strategy"][question_type]:
                                    stats["by_type_and_strategy"][question_type][tag] = {"correct": 0, "wrong": 0, "total": 0}
                                stats["by_type_and_strategy"][question_type][tag]["correct"] += 1
                                stats["by_type_and_strategy"][question_type][tag]["total"] += 1
                            elif label == "WRONG":
                                stats["wrong"] += 1
                                if question_type not in stats["by_type"]:
                                    stats["by_type"][question_type] = {"correct": 0, "wrong": 0, "total": 0}
                                stats["by_type"][question_type]["wrong"] += 1
                                stats["by_type"][question_type]["total"] += 1
                                if tag not in stats["by_strategy"]:
                                    stats["by_strategy"][tag] = {"correct": 0, "wrong": 0, "total": 0}
                                stats["by_strategy"][tag]["wrong"] += 1
                                stats["by_strategy"][tag]["total"] += 1
                                # Nested type×strategy
                                if question_type not in stats["by_type_and_strategy"]:
                                    stats["by_type_and_strategy"][question_type] = {}
                                if tag not in stats["by_type_and_strategy"][question_type]:
                                    stats["by_type_and_strategy"][question_type][tag] = {"correct": 0, "wrong": 0, "total": 0}
                                stats["by_type_and_strategy"][question_type][tag]["wrong"] += 1
                                stats["by_type_and_strategy"][question_type][tag]["total"] += 1

                    # Also collect latencies from the original result file
                    try:
                        with open(result_file, 'r') as rf:
                            result_data = json.load(rf)
                            if "error" not in result_data:
                                for item in result_data.get("retrieval_results", []):
                                    name = item.get("strategy", "default")
                                    tag = f"{name}_generated_context"
                                    latency_ms = item.get("latency")
                                    if latency_ms is not None:
                                        stats["latencies"].setdefault(tag, []).append(latency_ms / 1000)
                    except Exception:
                        pass

                except Exception as e:
                    print(f"[{filename}] Error loading existing evaluation: {e}")

                print(f"[{filename}] Already evaluated, skipping...")
                continue
        
        files_to_process.append(result_file)
    
    print(f"Processing {len(files_to_process)} files in parallel (rate limited to {int(60.0 / judge._min_interval)} requests/minute)")
    print("-" * 80)
    
    # Launch all files in parallel - rate limiter will throttle actual API calls
    tasks = [
        evaluate_single_file(
            result_file,
            provider,
            provider_output_dir,
            judge,
            log_lock,
            stats
        )
        for result_file in files_to_process
    ]
    
    # Execute all tasks - the rate limiter lock ensures API calls stay within limit
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Calculate accuracy
    total_labeled = stats["correct"] + stats["wrong"]
    if total_labeled > 0:
        stats["accuracy"] = stats["correct"] / total_labeled
    else:
        stats["accuracy"] = 0.0
    
    return stats


def _build_pass_at_k_tables(
    provider: str,
    benchmark_version: str,
    output_dir: Path,
    console: Console,
) -> None:
    """Read all evaluation files and display pass@k tables using rich.

    For each example+strategy, we check passes 1..K.  An example counts as
    pass@k-correct if *any* pass <= k was CORRECT.

    Tables shown:
      1. Per-strategy pass@k accuracy
      2. Per question-type pass@k accuracy (aggregated across strategies)
      3. Per question-type × strategy pass@k accuracy
    """
    eval_dir = output_dir / provider / "longmemeval" / benchmark_version
    if not eval_dir.exists():
        console.print(f"[yellow]No evaluation directory found: {eval_dir}[/yellow]")
        return

    # Collect: (index, question_type) -> {pass_idx -> {strategy_tag -> label}}
    data: Dict[tuple, Dict[int, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    max_pass = 0

    for eval_file in sorted(eval_dir.glob("*.json")):
        m = re.match(r'^(\d+)_(.+)_pass_(\d+)$', eval_file.stem)
        if not m:
            # legacy
            m2 = re.match(r'^(\d+)_(.+)$', eval_file.stem)
            if not m2:
                continue
            idx, qtype, pidx = int(m2.group(1)), m2.group(2), 1
        else:
            idx, qtype, pidx = int(m.group(1)), m.group(2), int(m.group(3))

        max_pass = max(max_pass, pidx)
        try:
            with open(eval_file, "r") as f:
                ed = json.load(f)
            for jr in ed.get("judge_response", []):
                tag = jr.get("tag", "generated_context")
                label = jr.get("label", "WRONG")
                data[(idx, qtype)][pidx][tag] = label
        except Exception:
            continue

    if not data:
        console.print("[yellow]No evaluation data found.[/yellow]")
        return

    # Determine all strategies and question types
    all_strategies = sorted({
        tag
        for passes in data.values()
        for tags in passes.values()
        for tag in tags
    })
    all_qtypes = sorted({qtype for _, qtype in data.keys()})

    def _short(tag: str) -> str:
        return tag.replace("_generated_context", "").replace("_", " ").title()

    # ── Table 1: Per-strategy pass@k ──
    table1 = Table(title=f"Pass@k Accuracy by Strategy — {provider} / {benchmark_version}", show_lines=True)
    table1.add_column("Strategy", style="bold")
    for k in range(1, max_pass + 1):
        table1.add_column(f"pass@{k}", justify="center")

    for strat in all_strategies:
        row_vals: list[str] = []
        # examples that have this strategy in at least one pass
        relevant = {key for key, passes in data.items() if any(strat in tags for tags in passes.values())}
        for k in range(1, max_pass + 1):
            correct = 0
            for key in relevant:
                # pass@k: correct if any pass 1..k was CORRECT for this strategy
                if any(data[key].get(p, {}).get(strat) == "CORRECT" for p in range(1, k + 1)):
                    correct += 1
            total = len(relevant)
            if total > 0:
                pct = correct / total * 100
                row_vals.append(f"{correct}/{total} ({pct:.1f}%)")
            else:
                row_vals.append("—")
        table1.add_row(_short(strat), *row_vals)

    console.print()
    console.print(table1)

    # ── Table 2: Per question-type × strategy pass@k (one table per strategy) ──
    for strat in all_strategies:
        table3 = Table(
            title=f"Pass@k by Question Type — {_short(strat)} — {provider} / {benchmark_version}",
            show_lines=True,
        )
        table3.add_column("Question Type", style="bold")
        for k in range(1, max_pass + 1):
            table3.add_column(f"pass@{k}", justify="center")

        for qtype in all_qtypes:
            relevant = {key for key in data if key[1] == qtype}
            row_vals = []
            for k in range(1, max_pass + 1):
                correct = 0
                # examples that have this strategy
                strat_relevant = {
                    key for key in relevant
                    if any(strat in data[key].get(p, {}) for p in range(1, max_pass + 1))
                }
                for key in strat_relevant:
                    if any(data[key].get(p, {}).get(strat) == "CORRECT" for p in range(1, k + 1)):
                        correct += 1
                total = len(strat_relevant)
                if total > 0:
                    pct = correct / total * 100
                    row_vals.append(f"{correct}/{total} ({pct:.1f}%)")
                else:
                    row_vals.append("—")
            table3.add_row(qtype, *row_vals)

        console.print()
        console.print(table3)

    # ── Latency table ──
    # Collect latencies from run files
    runs_dir = output_dir.parent / "runs" / provider / "longmemeval" / benchmark_version
    if runs_dir.exists():
        latencies: Dict[str, list] = defaultdict(list)
        for run_file in runs_dir.glob("*.json"):
            try:
                with open(run_file) as f:
                    rd = json.load(f)
                if "error" in rd:
                    continue
                for item in rd.get("retrieval_results", []):
                    name = item.get("strategy", "default")
                    tag = f"{name}_generated_context"
                    latency_ms = item.get("latency")
                    if latency_ms is not None:
                        latencies[tag].append(latency_ms / 1000)
            except Exception:
                continue

        if latencies:
            import numpy as np
            ltable = Table(title=f"Latency Statistics (seconds) — {provider} / {benchmark_version}", show_lines=True)
            ltable.add_column("Strategy", style="bold")
            for col in ("Min", "P25", "Median", "P95", "Max", "Count"):
                ltable.add_column(col, justify="right")
            for tag in sorted(latencies.keys()):
                arr = np.array(latencies[tag])
                ltable.add_row(
                    _short(tag),
                    f"{np.min(arr):.3f}",
                    f"{np.percentile(arr, 25):.3f}",
                    f"{np.percentile(arr, 50):.3f}",
                    f"{np.percentile(arr, 95):.3f}",
                    f"{np.max(arr):.3f}",
                    str(len(arr)),
                )
            console.print()
            console.print(ltable)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate LongMemEval runs using LLM-as-Judge with AWS Bedrock"
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["mem0", "nebula", "recallr", "supermemory", "zep"],
        help="Provider to evaluate"
    )
    parser.add_argument(
        "--benchmark-version",
        type=str,
        required=True,
        choices=["oracle", "medium", "small"],
        help="Benchmark version (oracle, medium, small)"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory for runs (default: runs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluations",
        help="Base directory for evaluations (default: evaluations)"
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=50,
        help="Rate limit: requests per minute (default: 50)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun even if evaluations exist"
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Skip evaluation and only display pass@k results from existing evaluation files"
    )
    
    args = parser.parse_args()
    console = Console()

    if not args.results_only:
        # Get API key from environment
        bedrock_api_key = os.getenv("BEDROCK_API_KEY")
        if not bedrock_api_key:
            parser.error("BEDROCK_API_KEY environment variable is required")
        
        # Initialize judge with hardcoded configuration
        judge = BedrockJudge(
            api_key=bedrock_api_key,
            region=BEDROCK_REGION,
            model=BEDROCK_MODEL,
            thinking_budget=THINKING_BUDGET
        )
        
        # Set rate limit
        judge.set_rate_limit(args.requests_per_minute)
        
        console.print("Bedrock Judge initialized:")
        console.print(f"  Model: {BEDROCK_MODEL}")
        console.print(f"  Region: {BEDROCK_REGION}")
        console.print(f"  Thinking budget: {THINKING_BUDGET}")
        console.print(f"  Rate limit: {args.requests_per_minute} requests/minute")

        # Run evaluation asynchronously
        stats = asyncio.run(run_evaluation(
            provider=args.provider,
            benchmark_version=args.benchmark_version,
            runs_dir=Path(args.runs_dir),
            output_dir=Path(args.output_dir),
            judge=judge,
            skip_existing=not args.force_rerun
        ))
        
        # Print run summary
        console.print()
        console.print(f"[bold]Evaluation Summary: {args.provider} / {args.benchmark_version}[/bold]")
        console.print(f"  Total result files: {stats.get('total', 0)}")
        console.print(f"  Successful results: {stats.get('successful_results', 0)}")
        console.print(f"  Failed results: {stats.get('failed_results', 0)}")
        console.print(f"  Evaluated (new): {stats.get('evaluated', 0)}")
        console.print(f"  Skipped (existing): {stats.get('skipped', 0)}")
        if stats.get('judge_failed', 0) > 0:
            console.print(f"  [red]Judge failures: {stats['judge_failed']}[/red]")

    # Always display pass@k results from evaluation files
    _build_pass_at_k_tables(
        provider=args.provider,
        benchmark_version=args.benchmark_version,
        output_dir=Path(args.output_dir),
        console=console,
    )


if __name__ == "__main__":
    main()
