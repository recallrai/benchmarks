"""
Run Supermemory on LongMemEval benchmark.

Directory structure:
    runs/supermemory/longmemeval/{version}/{index}_{question_type}.json

Environment variables (.env):
    SUPERMEMORY_API_KEY=your_api_key

Usage:
python3 run_supermemory_longmemeval.py \
    --data-path data/longmemeval/longmemeval_oracle.json \
    --start-index 0 --end-index 10 --output-dir runs
"""

import os
import time
import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from supermemory import AsyncSupermemory, RateLimitError

from base_runner import BaseLongMemEvalRunner

load_dotenv()


class SupermemoryRunner(BaseLongMemEvalRunner):
    provider_name = "supermemory"
    expected_strategies = ("default",)
    rate_limit_errors = (RateLimitError,)

    def __init__(
        self,
        api_key: str,
        data_path: str,
        output_base_dir: str,
        parallelism: int = 1,
    ):
        self.api_key = api_key
        super().__init__(data_path, output_base_dir, parallelism)

    # ---- Client lifecycle ----

    @asynccontextmanager
    async def create_client(self):
        yield AsyncSupermemory(api_key=self.api_key)

    # ---- Core logic ----

    async def run_single_example(self, index: int, example: Dict[str, Any], client: AsyncSupermemory, pass_idx: int) -> Dict[str, Any]:
        question_type = example["question_type"]
        haystack_sessions = example["haystack_sessions"]
        question = example["question"]
        ground_truth_answer = example["answer"]
        question_date = example["question_date"]

        container_tag = f"longmemeval_{self.version}_{index}_{question_type}"

        # Fetch existing documents
        try:
            existing_docs_resp = await self._with_retry(
                client.documents.list,
                container_tags=[container_tag],
                limit=len(haystack_sessions)
            )
            existing_memories = existing_docs_resp.memories
        except Exception:
            existing_memories = []

        # Process each haystack session and add as documents
        document_ids = []
        for s_idx, haystack_session in enumerate(haystack_sessions):
            
            # Check if this session is already processed
            existing_s_idx = [
                m for m in existing_memories
                if isinstance(m.metadata, dict) and m.metadata.get("session_index") == s_idx
            ]
            if existing_s_idx and existing_s_idx[0].status == "done":
                await self.log(f"[Example {index}] Session {s_idx} already processed, skipping")
                document_ids.append(existing_s_idx[0].id)
                continue

            await self.log(f"[Example {index}] Processing session {s_idx + 1}/{len(haystack_sessions)}")
            # Convert session messages to a formatted string
            content = "".join(
                f"{doc['role'].capitalize()}: {doc['content']}\n"
                for doc in haystack_session
            )
            # Add document to Supermemory
            resp = await self._with_retry(
                client.add,
                content=content,
                container_tag=container_tag,    # user_id
                metadata={
                    "benchmark_name": "longmemeval",
                    "benchmark_version": self.version,
                    "benchmark_index": index,
                    "session_type": "haystack",
                    "session_index": s_idx,
                    "question_type": question_type,
                },
            )
            document_ids.append(resp.id)
            await self.log(f"[Example {index}] Added document {resp.id} for session {s_idx}")
            
            # Wait for indexing to complete
            wait_start = time.perf_counter()
            await self.log(f"[Example {index}] Waiting for document {resp.id} to complete...")
            while True:
                doc = await self._with_retry(client.documents.get, resp.id)
                elapsed = time.perf_counter() - wait_start
                if doc.status == "done":
                    await self.log(f"[Example {index}] Document {resp.id}: completed in {elapsed:.1f}s")
                    break
                if doc.status == "failed":
                    raise RuntimeError(f"Document {resp.id} indexing failed")
                await self.log(
                    f"[Example {index}] Document {resp.id}: "
                    f"still processing [{elapsed:.1f}s] ({doc.status})"
                )
                await asyncio.sleep(2)

        # Skip strategies logic just like mem0/recallr
        skip_strategies = self.get_strategies_correct_in_any_pass(index, question_type, pass_idx - 1)
        
        retrieval_results = {}
        
        for strategy in self.expected_strategies:
            if strategy in skip_strategies:
                await self.log(f"[Example {index}] {strategy}: CORRECT in previous pass, skipping")
                continue

            await self.log(f"[Example {index}] Searching for context: {question}...")
            
            resp, latency = await self._with_retry_timed(
                client.search.memories,
                q=question,
                container_tag=container_tag,
                threshold=0.6,
                limit=20
            )
            
            results = resp.results
            context = "\n".join(r.memory for r in results)
            
            retrieval_results[strategy] = {
                "strategy": strategy,
                "context": context,
                "latency": latency,
            }
            await self.log(f"[Example {index}] Retrieved {len(results)} results in {latency / 1000:.2f}s")
            
        result_list = [retrieval_results[k] for k in self.expected_strategies if k in retrieval_results]

        return {
            "index": index,
            "question_type": question_type,
            "question": question,
            "question_date": question_date,
            "ground_truth_answer": str(ground_truth_answer),
            "metadata": {
                "supermemory_container_tag": container_tag,
                "supermemory_document_ids": document_ids,
            },
            "retrieval_results": result_list,
        }

    # ---- CLI ----

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "SupermemoryRunner":
        api_key = os.getenv("SUPERMEMORY_API_KEY")
        if not api_key:
            raise SystemExit("SUPERMEMORY_API_KEY environment variable is required")
        return cls(
            api_key=api_key,
            data_path=args.data_path,
            output_base_dir=args.output_dir,
            parallelism=args.parallelism,
        )


if __name__ == "__main__":
    SupermemoryRunner.main()
