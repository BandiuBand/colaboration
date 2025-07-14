from dataclasses import dataclass
from typing import Any, Dict, Optional

from .llm import LLM

@dataclass
class Step:
    """Container for instructions used to transform a dataset entry."""

    task_prompt: str
    validation_prompt: str
    selection_prompt: Optional[str] = None
    llm: LLM | None = None

    def execute(self, input: Dict[str, Any]) -> str:
        """Run ``task_prompt`` through the associated LLM and return the output."""
        if self.llm is None:
            raise RuntimeError("LLM instance is not set for this step")
        prompt = self.task_prompt.format(**input)
        return self.llm.process(prompt)

