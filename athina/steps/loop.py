import asyncio
from typing import Dict, List, Any, Optional
from athina.steps.base import Step
from pydantic import ConfigDict
from athina.steps.code_execution_v2 import CodeExecutionV2, EXECUTION_E2B
from concurrent.futures import ThreadPoolExecutor


class LoopStep(Step):
    """Step that evaluates conditions and executes appropriate branch steps."""

    loop_type: str
    loop_input: Optional[str]
    loop_count: Optional[int]
    sequence: List[Step]
    execution_mode: Optional[str]
    max_workers: int = 5  # Default number of workers

    async def _execute_single_step(self, step: Step, context: Dict) -> Dict:
        """Execute a single step asynchronously using ThreadPoolExecutor."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor,
                step.execute,
                context
            )

    async def _execute_steps_async(self, steps: List[Step], inputs: Dict, semaphore: asyncio.Semaphore) -> Dict:
        """Execute a sequence of steps with given inputs asynchronously."""
        async with semaphore:
            cumulative_context = inputs.copy()
            final_output = None
            executed_steps = []

            for step in steps:
                # Execute each step asynchronously
                step_result = await self._execute_single_step(step, cumulative_context)
                executed_steps.append(step_result)
                cumulative_context = {
                    **cumulative_context,
                    f"{step.name}": step_result.get("data", {}),
                }
                final_output = step_result.get("data")

            return {
                "status": "success",
                "data": final_output,
                "metadata": {"executed_steps": executed_steps},
            }

    async def _process_batch(self, batch_items: List[tuple], inputs: Dict, semaphore: asyncio.Semaphore) -> List[Dict]:
        """Process a batch of items in parallel."""
        tasks = []
        for index, item in batch_items:
            context = {**inputs, "item": item, "index": index, "count": len(batch_items)}
            tasks.append(self._execute_steps_async(self.sequence, context, semaphore))
        return await asyncio.gather(*tasks)
    
    async def _process_count(self, count: int, inputs: Dict, semaphore: asyncio.Semaphore) -> List[Dict]:
        """Process a count of items in parallel."""
        tasks = []
        for index in range(count):
            context = {**inputs, "index": index, "count": count}
            tasks.append(self._execute_steps_async(self.sequence, context, semaphore))
        return await asyncio.gather(*tasks)

    def execute(self, inputs: Dict) -> Dict:
        """Execute the loop step by running input list items on appropriate steps one by one."""
        try:
            if self.loop_type == "map":
                loop_input = inputs.get(self.loop_input, [])
                
                if not isinstance(loop_input, list):
                    return {
                        "status": "error",
                        "data": "Input not of type list",
                        "metadata": {},
                    }
                
                current = []
                executed_steps = []
                
                if self.execution_mode == "parallel":
                    async def run_parallel():
                        semaphore = asyncio.Semaphore(self.max_workers)
                        # Create a list of (index, item) tuples
                        indexed_items = list(enumerate(loop_input))
                        results = await self._process_batch(indexed_items, inputs, semaphore)
                        return results

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(run_parallel())
                        for result in results:
                            current.append(result.get("data"))
                            executed_steps.append(result.get("metadata"))
                    finally:
                        loop.close()
                else:
                    # Sequential execution
                    for index, item in enumerate(loop_input):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            semaphore = asyncio.Semaphore(1)
                            result = loop.run_until_complete(
                                self._execute_steps_async(
                                    self.sequence,
                                    {**inputs, "item": item, "index": index, "count": len(loop_input)},
                                    semaphore
                                )
                            )
                            current.append(result.get("data"))
                            executed_steps.append(result.get("metadata"))
                        finally:
                            loop.close()

                return {
                    "status": "success",
                    "data": current,
                    "metadata": {
                        "executed_steps": executed_steps,
                    },
                }
                
            else:
                count = self.loop_count
                
                if not isinstance(count, int) or count <= 0:
                    return {
                        "status": "error",
                        "data": "Invalid loop count",
                        "metadata": {},
                    }
                
                current = []
                executed_steps = []
                
                if self.execution_mode == "parallel":
                    async def run_parallel():
                        semaphore = asyncio.Semaphore(self.max_workers)
                        results = await self._process_count(count=count, inputs = inputs, semaphore = semaphore)
                        return results

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(run_parallel())
                        for result in results:
                            current.append(result.get("data"))
                            executed_steps.append(result.get("metadata"))
                    finally:
                        loop.close()
                else:
                    # Sequential execution
                    for index in range(count):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            semaphore = asyncio.Semaphore(1)
                            result = loop.run_until_complete(
                                self._execute_steps_async(
                                    self.sequence,
                                    {**inputs, "index": index, "count": count},
                                    semaphore
                                )
                            )
                            current.append(result.get("data"))
                            executed_steps.append(result.get("metadata"))
                        finally:
                            loop.close()

                return {
                    "status": "success",
                    "data": current,
                    "metadata": {
                        "executed_steps": executed_steps,
                    },
                }
                
        except Exception as e:
            return {
                "status": "error",
                "data": f"Loop step execution failed: {str(e)}",
                "metadata": {},
            }