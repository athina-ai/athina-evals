import time
from typing import List
from ..evals import BaseEvaluator
from .exception import AthinaGuardException
from concurrent.futures import ThreadPoolExecutor, as_completed


def guard(suite: List[BaseEvaluator], **kwargs):
    # Define the maximum number of threads to use
    max_workers = 10  # Adjust based on your needs and environment
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation functions to the executor
        future_to_eval = {executor.submit(eval.guard, **kwargs): eval for eval in suite}

        for future in as_completed(future_to_eval):
            eval = future_to_eval[future]
            try:
                guard_result = future.result()
                passed = guard_result.passed
                reason = guard_result.reason
                runtime = guard_result.runtime
                if passed:
                    print(f"{eval.display_name}: Passed in {runtime}ms - {reason}")
                else:
                    print(f"{eval.display_name}: Failed in {runtime}ms - {reason}")
                    raise AthinaGuardException(f"{eval.display_name} failed: {reason}")
            except Exception as exc:
                raise exc

    end_time = time.perf_counter()
    response_time_ms = (end_time - start_time) * 1000
    print(f"Guard completed in {response_time_ms}ms")
