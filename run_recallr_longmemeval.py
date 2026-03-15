"""
Run Recallr AI on LongMemEval benchmark.

Directory structure:
    runs/recallr/longmemeval/{version}/{index}_{question_type}.json

Environment variables (.env):
    RECALLR_PROJECT_ID=your_project_id
    RECALLR_API_KEY=your_api_key

Usage:
python3 run_recallr_longmemeval.py \
    --data-path data/longmemeval/longmemeval_oracle.json \
    --start-index 0 --end-index 25 --parallelism 5 --output-dir runs
"""

import os
import time
import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from recallrai import AsyncRecallrAI
from recallrai.models import MessageRole, SessionStatus, RecallStrategy
from recallrai.exceptions import UserNotFoundError, RateLimitError

from base_runner import BaseLongMemEvalRunner

load_dotenv()


class RecallrRunner(BaseLongMemEvalRunner):
    provider_name = "recallr"
    expected_strategies = ("low_latency", "balanced", "agentic")
    rate_limit_errors = (RateLimitError,)

    def __init__(
        self,
        api_key: str,
        project_id: str,
        data_path: str,
        output_base_dir: str,
        parallelism: int = 1,
    ):
        self.api_key = api_key
        self.project_id = project_id
        super().__init__(data_path, output_base_dir, parallelism)

    # ---- Client lifecycle ----

    @asynccontextmanager
    async def create_client(self):
        async with AsyncRecallrAI(api_key=self.api_key, project_id=self.project_id, timeout=300) as client:
            yield client

    # ---- Core logic ----

    async def run_single_example(self, index: int, example: Dict[str, Any], client: AsyncRecallrAI, pass_idx: int) -> Dict[str, Any]:
        question_type = example["question_type"]
        haystack_sessions = example["haystack_sessions"]
        haystack_dates = example["haystack_dates"]
        question = example["question"]
        ground_truth_answer = example["answer"]
        question_date = example["question_date"]

        user_id = f"longmemeval_{self.version}_{index}_{question_type}"

        # Get or create user
        try:
            user = await self._with_retry(client.get_user, user_id)
            await self.log(f"[Example {index}] Found existing user: {user_id}")

        except UserNotFoundError:
            user = await self._with_retry(
                client.create_user,
                user_id=user_id,
                metadata={
                    "benchmark_name": "longmemeval",
                    "benchmark_version": self.version,
                    "benchmark_index": index,
                    "question_type": question_type,
                },
            )
            await self.log(f"[Example {index}] Created new user: {user_id}")

        # Ingest haystack sessions
        session_ids = []

        sessions = await self._with_retry(
            user.list_sessions,
            offset=0,
            limit=len(haystack_sessions),
            metadata_filter={"session_type": "haystack"},
            status_filter=[SessionStatus.PROCESSED],
        )

        await self.log(
            f"[Example {index}] Existing sessions: {sessions.total}, "
            f"Expected: {len(haystack_sessions)}"
        )

        if sessions.total == len(haystack_sessions) and all(
            s.status == SessionStatus.PROCESSED for s in sessions.sessions
        ):
            await self.log(
                f"[Example {index}] All haystack sessions already processed"
            )
            session_ids = [s.session_id for s in sessions.sessions]

        else:
            for s_idx, haystack_session in enumerate(haystack_sessions):
                await self.log(
                    f"[Example {index}] Processing haystack session "
                    f"{s_idx + 1}/{len(haystack_sessions)}"
                )
                existing = await self._with_retry(
                    user.list_sessions,
                    offset=0,
                    limit=1,
                    metadata_filter={
                        "session_type": "haystack",
                        "haystack_index": s_idx,
                    },
                    status_filter=[SessionStatus.PROCESSED],
                )
                if existing.total == 1:
                    await self.log(
                        f"[Example {index}] Session {s_idx} already processed, skipping"
                    )
                    session_ids.append(existing.sessions[0].session_id)
                    continue

                haystack_dt = self.parse_longmemeval_benchmark_datetime(haystack_dates[s_idx])
                session = await self._with_retry(
                    user.create_session,
                    auto_process_after_seconds=600,
                    custom_created_at_utc=haystack_dt,
                    metadata={
                        "session_type": "haystack",
                        "haystack_index": s_idx,
                        "original_datetime": haystack_dates[s_idx],
                    },
                )
                session_ids.append(session.session_id)

                for doc in haystack_session:
                    await self._with_retry(
                        session.add_message,
                        role=MessageRole(doc["role"]),
                        content=doc["content"]
                    )
                await self._with_retry(session.process)

                wait_start = time.perf_counter()
                sid = session.session_id
                await self.log(f"[Example {index}] Waiting for session {sid} to complete...")
                while True:
                    await self._with_retry(session.refresh)
                    elapsed = time.perf_counter() - wait_start
                    if session.status == SessionStatus.PROCESSED:
                        await self.log(f"[Example {index}] Session {sid}: completed in {elapsed:.1f}s")
                        break
                    if session.status == SessionStatus.FAILED:
                        raise RuntimeError(f"Session {sid} processing failed")
                    await self.log(
                        f"[Example {index}] Session {sid}: "
                        f"still processing [{elapsed:.1f}s] ({session.status})"
                    )
                    await asyncio.sleep(10)

        # Create query session
        question_dt = self.parse_longmemeval_benchmark_datetime(question_date)

        await self.log(f"[Example {index}] Creating query session for: {question}")

        query_session = await self._with_retry(
            user.create_session,
            auto_process_after_seconds=600,
            custom_created_at_utc=question_dt,
            metadata={
                "session_type": "query",
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "original_datetime": question_date,
            },
        )
        await self._with_retry(
            query_session.add_message,
            role=MessageRole.USER, content=question
        )

        # Retrieve context with all recall strategies
        skip_strategies = self.get_strategies_correct_in_any_pass(index, question_type, pass_idx - 1)
        
        retrieval_results: Dict[str, Any] = {}
        
        for strategy in self.expected_strategies:
            if strategy in skip_strategies:
                await self.log(f"[Example {index}] {strategy}: CORRECT in previous pass, skipping")
                continue

            await self.log(f"[Example {index}] Retrieving with {strategy} strategy...")
            ctx, latency = await self._with_retry_timed(
                query_session.get_context,
                min_top_k=20, max_top_k=20,
                memories_threshold=0.6,
                recall_strategy=RecallStrategy(strategy)
            )

            retrieval_results[strategy] = {
                "strategy": strategy,
                "context": ctx.context,
                "latency": latency,
            }
            await self.log(
                f"[Example {index}] {strategy}: retrieved in {latency:.2f}ms"
            )

        if query_session.status in (SessionStatus.PENDING, SessionStatus.FAILED):
            await self._with_retry(query_session.process)

        result_list = [retrieval_results[k] for k in self.expected_strategies if k in retrieval_results]

        return {
            "index": index,
            "question_type": question_type,
            "question": question,
            "question_date": question_date,
            "ground_truth_answer": str(ground_truth_answer),
            "metadata": {
                "recallr_user_id": user.user_id,
                "recallr_haystack_session_ids": session_ids,
                "recallr_query_session_id": query_session.session_id,
            },
            "retrieval_results": result_list,
        }

    # ---- CLI ----

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "RecallrRunner":
        api_key = os.getenv("RECALLR_API_KEY")
        project_id = os.getenv("RECALLR_PROJECT_ID")
        if not api_key:
            raise SystemExit("RECALLR_API_KEY environment variable is required")
        if not project_id:
            raise SystemExit("RECALLR_PROJECT_ID environment variable is required")
        return cls(
            api_key=api_key,
            project_id=project_id,
            data_path=args.data_path,
            output_base_dir=args.output_dir,
            parallelism=args.parallelism,
        )


if __name__ == "__main__":
    RecallrRunner.main()
