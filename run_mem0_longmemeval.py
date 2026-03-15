"""
Run Mem0 on LongMemEval benchmark.

Directory structure:
    runs/mem0/longmemeval/{version}/{index}_{question_type}.json

Environment variables (.env):
    MEM0_API_KEY=your_api_key

Usage:
python3 run_mem0_longmemeval.py \
    --data-path data/longmemeval/longmemeval_oracle.json \
    --start-index 0 --end-index 10 --parallelism 5 --output-dir runs
"""

import os
import argparse
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from mem0 import AsyncMemoryClient
from mem0.exceptions import RateLimitError

from base_runner import BaseLongMemEvalRunner

load_dotenv()


class Mem0Runner(BaseLongMemEvalRunner):
    provider_name = "mem0"
    expected_strategies = ("non_graph", "graph")
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
        async with AsyncMemoryClient(api_key=self.api_key) as client:
            yield client

    # ---- Core logic ----
    async def run_single_example(self, index: int, example: Dict[str, Any], client: AsyncMemoryClient, pass_idx: int) -> Dict[str, Any]:
        question_type = example["question_type"]
        haystack_sessions = example["haystack_sessions"]
        haystack_dates = example["haystack_dates"]
        question = example["question"]
        ground_truth_answer = example["answer"]
        question_date = example["question_date"]

        user_id = f"longmemeval_{self.version}_{index}_{question_type}"
        await self.log(f"[Example {index}] Processing memories for user: {user_id}")

        # Check existing memories to avoid re-ingesting already processed sessions
        response = await self._with_retry(
            client.get_all, filters={"user_id": user_id}, version="v2"
        )
        existing_memories = response["results"]

        processed_haystack_indices = set()
        for mem in existing_memories:
            metadata = mem["metadata"]
            if "haystack_index" in metadata:
                processed_haystack_indices.add(metadata["haystack_index"])

        # Process each haystack session and add memories synchronously
        for s_idx, haystack_session in enumerate(haystack_sessions):
            if s_idx in processed_haystack_indices:
                await self.log(f"[Example {index}] Session {s_idx + 1}/{len(haystack_sessions)} already processed, skipping")
                continue

            await self.log(
                f"[Example {index}] Processing session "
                f"{s_idx + 1}/{len(haystack_sessions)}"
            )
            # Parse the datetime for this haystack session
            haystack_dt = self.parse_longmemeval_benchmark_datetime(haystack_dates[s_idx])
            # Convert datetime to Unix timestamp for mem0
            timestamp_unix = int(haystack_dt.timestamp())

            # Convert messages to mem0 format
            messages = [
                {"role": doc["role"], "content": doc["content"]}
                for doc in haystack_session
            ]

            # Add memories synchronously using async_mode=False
            # This ensures memories are ingested in real-time
            await self._with_retry(
                client.add,
                messages=messages,
                user_id=user_id,
                metadata={
                    "benchmark_name": "longmemeval",
                    "benchmark_version": self.version,
                    "benchmark_index": index,
                    "question_type": question_type,
                    "session_type": "haystack",
                    "haystack_index": s_idx,
                    "original_datetime": haystack_dates[s_idx],
                },
                timestamp=timestamp_unix,  # Custom timestamp for the session
                async_mode=False,  # Synchronous mode - no waiting needed!
                version="v2",
            )

        # Try both retrieval strategies: non-graph and graph
        skip_strategies = self.get_strategies_correct_in_any_pass(index, question_type, pass_idx - 1)

        retrieval_results = {}
        for strategy in self.expected_strategies:
            if strategy in skip_strategies:
                await self.log(f"[Example {index}] {strategy}: CORRECT in previous pass, skipping")
                continue

            await self.log(f"[Example {index}] Retrieving with {strategy} strategy...")
            raw, latency = await self._with_retry_timed(
                client.search,
                query=question,
                filters={"user_id": user_id},
                enable_graph=strategy == "graph",
            )

            memories = []
            for r in raw["results"]:
                memories.append(r["memory"])

            retrieval_results[strategy] = {
                "strategy": strategy,
                "context": "\n".join(memories),
                "latency": latency,
            }
            await self.log(
                f"[Example {index}] {strategy}: {len(memories)} memories "
                f"in {latency / 1000:.2f}s"
            )

        result_list = [retrieval_results[k] for k in self.expected_strategies if k in retrieval_results]

        return {
            "index": index,
            "question_type": question_type,
            "question": question,
            "question_date": question_date,
            "ground_truth_answer": str(ground_truth_answer),
            "metadata": {
                "mem0_user_id": user_id,
            },
            "retrieval_results": result_list,
        }

    # ---- CLI ----

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "Mem0Runner":
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise SystemExit("MEM0_API_KEY environment variable is required")
        return cls(
            api_key=api_key,
            data_path=args.data_path,
            output_base_dir=args.output_dir,
            parallelism=args.parallelism,
        )


if __name__ == "__main__":
    Mem0Runner.main()
