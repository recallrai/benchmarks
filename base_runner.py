"""
Base class for LongMemEval benchmark runners.
"""

import json
import time
import asyncio
import argparse
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar

_T = TypeVar("_T")


class BaseLongMemEvalRunner(ABC):
    """Base class for all LongMemEval benchmark runners."""

    provider_name: str  # Subclasses must set this as a class variable
    expected_strategies: Tuple  # Subclasses must set this
    rate_limit_errors: Tuple = ()  # Override in subclasses: e.g. (RateLimitError,)

    def __init__(self, data_path: str, output_base_dir: str, parallelism: int):
        self.parallelism = parallelism
        self._log_lock = asyncio.Lock()

        # Determine benchmark version from filename
        filename = Path(data_path).stem
        if "oracle" in filename.lower():
            self.version = "oracle"
        elif "_m" in filename.lower():
            self.version = "medium"
        elif "_s" in filename.lower():
            self.version = "small"
        else:
            raise ValueError(
                f"Cannot determine benchmark version from filename: {filename}"
            )

        # Set up output directory: runs/<provider>/longmemeval/<version>/
        self.output_dir = (
            Path(output_base_dir) / self.provider_name / "longmemeval" / self.version
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._result_cache: Dict[Tuple, Optional[Dict[str, Any]]] = {}
        self._eval_cache: Dict[Tuple, Optional[Dict[str, bool]]] = {}

        # Load benchmark data
        with open(data_path, "r", encoding="utf-8") as f:
            self.data =  json.load(f)

    # ---- Common utilities ----

    @staticmethod
    def parse_longmemeval_benchmark_datetime(date_str: str) -> datetime:
        """Parse LongMemEval datetime string to timezone-aware UTC datetime.

        Args:
            date_str: e.g. "2023/04/10 (Mon) 23:07"

        Returns:
            datetime with tzinfo=UTC
        """
        # "2023/04/10 (Mon) 23:07" -> "2023/04/10 23:07"
        date_str_clean = date_str.split(" (")[0] + " " + date_str.split(") ")[1]
        dt = datetime.strptime(date_str_clean, "%Y/%m/%d %H:%M")
        return dt.replace(tzinfo=timezone.utc)

    async def log(self, message: str) -> None:
        """Thread-safe print for parallel execution."""
        async with self._log_lock:
            print(message)

    # ---- Rate-limit retry helpers ----

    async def _with_retry(self, coro_fn: Callable[..., Coroutine[Any, Any, _T]], *args: Any, **kwargs: Any) -> _T:
        """Call an async function, retrying on rate-limit errors with exponential backoff.

        Retries indefinitely when the exception type matches any entry in
        ``self.rate_limit_errors``. Falls through immediately if
        ``rate_limit_errors`` is empty (the default).
        """
        delay = 2.0
        while True:
            try:
                return await coro_fn(*args, **kwargs)
            except Exception as exc:
                if self.rate_limit_errors and isinstance(exc, self.rate_limit_errors):
                    await self.log(f"Rate limited, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise

    async def _with_retry_timed(self, coro_fn: Callable[..., Coroutine[Any, Any, _T]], *args: Any, **kwargs: Any) -> tuple[_T, float]:
        """Like _with_retry, but returns (result, latency_ms) measuring only the successful call."""
        delay = 2.0
        while True:
            try:
                start = time.perf_counter()
                result = await coro_fn(*args, **kwargs)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return result, latency_ms
            except Exception as exc:
                if self.rate_limit_errors and isinstance(exc, self.rate_limit_errors):
                    await self.log(f"Rate limited, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise

    # ---- Result file management ----

    def _get_result_path(self, index: int, question_type: str, pass_idx: int) -> Path:
        return self.output_dir / f"{index}_{question_type}_pass_{pass_idx}.json"

    def _result_exists(self, index: int, question_type: str, pass_idx: int) -> bool:
        return self._get_result_path(index, question_type, pass_idx).exists()

    def _load_result(self, index: int, question_type: str, pass_idx: int) -> Optional[Dict[str, Any]]:
        key = (index, question_type, pass_idx)
        if key not in self._result_cache:
            result_path = self._get_result_path(index, question_type, pass_idx)
            if result_path.exists():
                with open(result_path, "r", encoding="utf-8") as f:
                    self._result_cache[key] = json.load(f)
            else:
                self._result_cache[key] = None
        return self._result_cache[key]

    async def _save_result(self, index: int, question_type: str, pass_idx: int, result: Dict[str, Any]) -> None:
        result_path = self._get_result_path(index, question_type, pass_idx)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, indent=4, fp=f)
        await self.log(f"Saved result to: {result_path}")

    def get_strategies_correct_in_any_pass(self, index: int, question_type: str, up_to_pass: int) -> set[str]:
        """Collect strategies marked CORRECT across ALL passes from 1 to up_to_pass (inclusive).

        Returns:
            A set of strategy names that were CORRECT in any evaluated pass.
            Returns an empty set if no passes have been evaluated yet.
        """
        correct: set[str] = set()
        for p in range(1, up_to_pass + 1):
            eval_results = self.get_pass_evaluation(index, question_type, p)
            if eval_results is None:
                break  # No more evaluated passes beyond this point
            correct |= {name for name, is_correct in eval_results.items() if is_correct}
        return correct

    def get_strategies_needed_for_pass(self, index: int, question_type: str, pass_idx: int) -> set[str]:
        """Determine which strategies still need to be run for a given pass.

        For pass 1, all expected_strategies are needed.
        For pass N>1, only strategies that were NOT correct in ANY of passes 1..N-1.
        """
        if pass_idx == 1:
            return set(self.expected_strategies)
        correct = self.get_strategies_correct_in_any_pass(index, question_type, pass_idx - 1)
        return set(self.expected_strategies) - correct

    def is_successful_result(self, index: int, question_type: str, pass_idx: int) -> bool:
        """Check if a saved result contains all required strategies for this pass."""
        result = self._load_result(index, question_type, pass_idx)
        if not result:
            return False
        present = {item["strategy"] for item in result.get("retrieval_results", [])}
        needed = self.get_strategies_needed_for_pass(index, question_type, pass_idx)
        return needed.issubset(present)

    def describe_incomplete_result(self, index: int, question_type: str, pass_idx: int) -> str:
        """Human-readable explanation of why a saved result is not complete."""
        result = self._load_result(index, question_type, pass_idx)
        if not result:
            return "no result file found"
        present = {item["strategy"] for item in result.get("retrieval_results", [])}
        needed = self.get_strategies_needed_for_pass(index, question_type, pass_idx)
        missing = needed - present
        if missing:
            return f"missing retrieval strategies: {', '.join(sorted(missing))}"
        return "incomplete for unknown reason"

    # ---- Pass@k evaluation helpers ----

    def _get_eval_path(self, index: int, question_type: str, pass_idx: int) -> Path:
        """Return the evaluation file path for a given pass."""
        eval_dir = (
            Path(self.output_dir).parents[3]
            / "evaluations"
            / self.provider_name
            / "longmemeval"
            / self.version
        )
        return eval_dir / f"{index}_{question_type}_pass_{pass_idx}.json"

    def get_pass_evaluation(self, index: int, question_type: str, pass_idx: int) -> Optional[Dict[str, bool]]:
        """Check the evaluation file for a given pass.
        
        Returns:
            A dictionary mapping `strategy_name -> bool (is_correct)`.
            If the pass has not been evaluated yet, returns None.
        """
        key = (index, question_type, pass_idx)
        if key in self._eval_cache:
            return self._eval_cache[key]

        eval_path = self._get_eval_path(index, question_type, pass_idx)
        if not eval_path.exists():
            self._eval_cache[key] = None
            return None
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            responses = eval_data.get("judge_response", [])
            if not responses:
                self._eval_cache[key] = None
                return None
                
            eval_results = {}
            for r in responses:
                tag = r.get("tag", "").removesuffix("_generated_context")
                eval_results[tag] = (r.get("label") == "CORRECT")
                
            self._eval_cache[key] = eval_results
            return eval_results
        except Exception:
            self._eval_cache[key] = None
            return None

    # ---- Abstract methods ----

    @abstractmethod
    @asynccontextmanager
    async def create_client(self) -> AsyncIterator[Any]:
        """Create and yield the provider client.

        Example::

            @asynccontextmanager
            async def create_client(self):
                async with AsyncFoo(api_key=self.api_key) as client:
                    yield client
        """

    @abstractmethod
    async def run_single_example(self, index: int, example: Dict[str, Any], client: Any, pass_idx: int) -> Dict[str, Any]:
        """Run one benchmark example end-to-end.

        Args:
            index: Example index in the dataset.
            example: The benchmark example dict.
            client: Provider client instance.
            pass_idx: Current pass number (1-based). For multi-strategy
                providers, use get_pass_evaluation() for the previous pass
                to skip strategies that already passed.

        Must return a result dict containing at minimum::

            {
                "index": int,
                "question_type": str,
                "question": str,
                "question_date": str,
                "ground_truth_answer": str,
                "metadata": { ... },             # provider-specific IDs
                "retrieval_results": [
                    {"strategy": str, "context": str, "latency": float},
                    ...
                ],
            }
        """

    # ---- CLI hooks ----
    @classmethod
    @abstractmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "BaseLongMemEvalRunner":
        """Construct a runner instance from parsed CLI args."""

    # ---- Orchestration ----

    async def _process_example_wrapper(
        self,
        i: int,
        example: Dict[str, Any],
        start_index: int,
        total_examples: int,
        client: Any,
        pass_idx: int,
    ) -> None:
        """Run one example with error handling and immediate save."""
        question_type = example["question_type"]
        try:
            await self.log(
                f"\n[{i - start_index + 1}/{total_examples}] "
                f"Example {i} ({question_type}) [pass {pass_idx}]"
            )
            result = await self.run_single_example(i, example, client, pass_idx=pass_idx)
            await self._save_result(i, question_type, pass_idx, result)
            await self.log(f"[Example {i}] Successfully completed (pass {pass_idx})")
        except Exception as e:
            await self.log(f"[Example {i}] ERROR (pass {pass_idx}): {e}")

    async def run_all(
        self,
        start_index: int,
        end_index: Optional[int],
        max_passes: int,
    ) -> None:
        """Run benchmark examples with parallel execution and pass@k support."""
        if end_index is None:
            end_index = len(self.data) - 1

        total_examples = end_index - start_index + 1
        skipped = 0
        awaiting_eval = 0
        processed = 0
        failed = 0

        await self.log(f"Running {self.provider_name} on LongMemEval ({self.version})")
        await self.log(
            f"Processing examples {start_index} to {end_index} "
            f"({total_examples} total)"
        )
        await self.log(f"Max passes: {max_passes}")
        await self.log(f"Parallelism: {self.parallelism}")
        await self.log(f"Output directory: {self.output_dir}")
        await self.log("-" * 80)

        # Collect (index, example, pass_idx) tuples that need processing
        examples_to_process: List[Tuple[int, Dict[str, Any], int]] = []

        for i in range(start_index, end_index + 1):
            example = self.data[i]
            question_type = example["question_type"]

            # Determine which pass to run next
            next_pass = None

            for p in range(1, max_passes + 1):
                if not self.is_successful_result(i, question_type, p):
                    # Pass has no successful result – need to run it
                    next_pass = p
                    break
                
                # Run exists and is successful – check evaluation
                eval_results = self.get_pass_evaluation(i, question_type, p)
                if eval_results is None:
                    # Not yet evaluated – user needs to run evaluate_runs.py
                    awaiting_eval += 1
                    await self.log(
                        f"Example {i} ({question_type}): "
                        f"pass {p} awaiting evaluation, skipping..."
                    )
                    break
                
                # Check if ALL expected strategies are correct across all passes so far
                correct_so_far = self.get_strategies_correct_in_any_pass(i, question_type, p)
                if set(self.expected_strategies).issubset(correct_so_far):
                    # All strategies CORRECT – no more passes needed
                    processed += 1
                    await self.log(
                        f"Example {i} ({question_type}): "
                        "already CORRECT, skipping..."
                    )
                    break
                
                # At least one strategy still WRONG – continue to next pass

            if next_pass is None:
                skipped += 1
                continue

            # Additional log if it's incomplete
            reason = self.describe_incomplete_result(i, question_type, next_pass)
            await self.log(
                f"Example {i} ({question_type}): "
                f"pass {next_pass} incomplete/forced rerun ({reason}), rerunning..."
            )

            examples_to_process.append((i, example, next_pass))

        if awaiting_eval > 0:
            await self.log(
                f"\n*** {awaiting_eval} example(s) are awaiting evaluation. "
                f"Run evaluate_runs.py first, then re-run this script. ***\n"
            )

        # Process with shared client and semaphore-based parallelism
        async with self.create_client() as client:
            semaphore = asyncio.Semaphore(self.parallelism)

            async def _guarded(i: int, ex: Dict[str, Any], p: int) -> None:
                async with semaphore:
                    await self._process_example_wrapper(
                        i, ex, start_index, total_examples, client, pass_idx=p
                    )

            tasks = [_guarded(i, ex, p) for i, ex, p in examples_to_process]
            await asyncio.gather(*tasks)


        await self.log("\n" + "=" * 80)
        await self.log("SUMMARY")
        await self.log("=" * 80)
        await self.log(f"Total examples: {total_examples}")
        await self.log(f"Processed: {processed}")
        await self.log(f"Skipped: {skipped}")
        if awaiting_eval > 0:
            await self.log(f"Awaiting evaluation: {awaiting_eval}")
        await self.log(f"Failed: {failed}")
        await self.log(f"Output directory: {self.output_dir}")

    # ---- CLI entry point ----

    @classmethod
    def main(cls) -> None:
        """Standard CLI entry point. Call from ``if __name__ == '__main__'``."""
        parser = argparse.ArgumentParser(
            description=f"Run {cls.provider_name} on LongMemEval benchmark"
        )
        parser.add_argument(
            "--data-path",
            type=str,
            required=True,
            help="Path to longmemeval JSON file "
            "(e.g., data/longmemeval/longmemeval_oracle.json)",
        )
        parser.add_argument(
            "--start-index", type=int, default=0, help="Start index (default: 0)"
        )
        parser.add_argument(
            "--end-index",
            type=int,
            default=None,
            help="End index (inclusive). If not specified, runs to end of dataset.",
        )
        parser.add_argument(
            "--run-single",
            type=int,
            default=None,
            help="Run only a single example by index "
            "(overrides --start-index / --end-index)",
        )
        parser.add_argument(
            "--max-passes",
            type=int,
            default=1,
            help="Maximum number of passes per example (default: 1). "
            "When > 1, re-runs examples whose previous pass was evaluated "
            "as WRONG by evaluate_runs.py.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="runs",
            help="Base output directory (default: runs)",
        )
        parser.add_argument(
            "--parallelism",
            type=int,
            default=1,
            help="Number of examples to process in parallel (default: 1)",
        )

        args = parser.parse_args()
        runner = cls.from_cli_args(args)

        if args.run_single is not None:
            start_index = args.run_single
            end_index = args.run_single
        else:
            start_index = args.start_index
            end_index = args.end_index

        asyncio.run(
            runner.run_all(
                start_index=start_index,
                end_index=end_index,
                max_passes=args.max_passes,
            )
        )
