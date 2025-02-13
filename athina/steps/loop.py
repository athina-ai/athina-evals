import asyncio
import json
from typing import Dict, List, Any, Optional
from athina.steps.base import Step
from concurrent.futures import ThreadPoolExecutor
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined
from athina.helpers.step_helper import StepHelper

class Loop(Step):
    loop_type: str
    loop_input: Optional[str]
    loop_count: Optional[int]
    sequence: List[Step]
    execution_mode: Optional[str]
    max_workers: int = 5

    async def _execute_single_step(self, step: Step, context: Dict) -> Dict:
        """Execute a single step asynchronously using ThreadPoolExecutor."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor,
                step.execute,
                context
            )

    async def _execute_sequence(self, inputs: Dict, semaphore: asyncio.Semaphore) -> Dict:
        """Execute a sequence of steps asynchronously with proper context handling."""
        async with semaphore:
            context = inputs.copy()
            executed_steps = []
            final_output = None

            for step in self.sequence:
                result = await self._execute_single_step(step, context)
                executed_steps.append(result)
                context = {
                    **context,
                    f"{step.name}": result.get("data", {}),
                }
                final_output = result.get("data")  # Ensure final output is correctly captured

            return {
                "status": "success",
                "data": final_output,  # Ensure only final result is returned
                "metadata": {"executed_steps": executed_steps}
            }

    async def _execute_loop(self, inputs: Dict) -> Dict:
        """Handles loop execution, managing parallelism properly."""
        semaphore = asyncio.Semaphore(self.max_workers if self.execution_mode == "parallel" else 1)
        results = []

        if self.loop_type == "map":
            env = Environment(
                variable_start_string="{{",
                variable_end_string="}}",
                undefined=PreserveUndefined,
            )
            
            loop_input_template = env.from_string(self.loop_input)
            prepared_input_data = StepHelper.prepare_input_data(inputs)
            loop_input = loop_input_template.render(**prepared_input_data)
            items = json.loads(loop_input, strict=False) if loop_input else None
            if not isinstance(items, list):
                return {"status": "error", "data": "Input not of type list", "metadata": {}}

            tasks = [
                self._execute_sequence(
                    {**inputs, "item": item, "index": idx, "count": len(items)},
                    semaphore
                )
                for idx, item in enumerate(items)
            ]
        else:
            if not isinstance(self.loop_count, int) or self.loop_count <= 0:
                return {"status": "error", "data": "Invalid loop count", "metadata": {}}

            tasks = [
                self._execute_sequence(
                    {**inputs, "index": i, "count": self.loop_count},
                    semaphore
                )
                for i in range(self.loop_count)
            ]

        results = await asyncio.gather(*tasks)  # Gather results concurrently

        return {
            "status": "success",
            "data": [r["data"] for r in results],  # Ensure correct final output format
            "metadata": {"executed_steps": [r["metadata"] for r in results]}
        }

    def execute(self, inputs: Dict) -> Dict:
        """Handles execution, avoiding issues with already running event loops."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.ensure_future(self._execute_loop(inputs))
                loop.run_until_complete(future)
                return future.result()
            else:
                return asyncio.run(self._execute_loop(inputs))
        except Exception as e:
            return {"status": "error", "data": str(e), "metadata": {}}
