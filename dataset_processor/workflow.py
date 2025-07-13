from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .llm import LLM
from .step import Step


class Workflow:
    """Create dataset variations through a sequence of ``Step`` operations."""

    def __init__(
        self,
        dataset: Iterable[Dict[str, Any]],
        llm: LLM | None = None,
        num_variations: int = 50,
    ) -> None:
        self.dataset = list(dataset)
        self.llm = llm or LLM()
        self.num_variations = num_variations
        self.steps = self._build_steps()

    def _build_steps(self) -> List[Step]:
        """Return the default steps for the workflow."""
        step_gather = Step(
            task_prompt=(
                "Fill in any missing information required to solve the task:\n{prompt}"
            ),
            validation_prompt="Check that all necessary data is present.",
            llm=self.llm,
        )
        step_randomize = Step(
            task_prompt=(
                "Replace the facts in the following text with random alternatives:\n{prompt}"
            ),
            validation_prompt="Verify the facts were randomized.",
            llm=self.llm,
        )
        return [step_gather, step_randomize]

    def run(self) -> List[Dict[str, Any]]:
        """Execute the workflow and return the augmented dataset."""
        results: List[Dict[str, Any]] = []
        for entry in self.dataset:
            initial_prompt = self.steps[0].execute(entry)
            for _ in range(self.num_variations):
                new_prompt = self.steps[1].execute({"prompt": initial_prompt})
                results.append({"prompt": new_prompt})
        return results
