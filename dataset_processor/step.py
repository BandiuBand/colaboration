from dataclasses import dataclass
from typing import Optional

@dataclass
class Step:
    """Container for instructions used to transform a dataset entry."""

    task_prompt: str
    validation_prompt: str
    selection_prompt: Optional[str] = None
