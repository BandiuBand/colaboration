from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .step import Step


class DatasetProcessor:
    """Applies a sequence of ``Step`` instructions to each entry in a dataset."""

    def __init__(self, dataset: Iterable[Dict[str, Any]], steps: List[Step]):
        self.dataset = list(dataset)
        self.steps = steps

    def process(self) -> List[Dict[str, Any]]:
        """Return a new dataset built by applying ``steps`` to every item."""
        new_dataset: List[Dict[str, Any]] = []
        for entry in self.dataset:
            transformed = self._apply_steps(entry)
            new_dataset.extend(transformed)
        return new_dataset

    def _apply_steps(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a list of new entries derived from ``entry`` by each step."""
        results: List[Dict[str, Any]] = []
        for step in self.steps:
            item = entry.copy()
            item["task_prompt"] = step.task_prompt.format(**entry)
            item["validation_prompt"] = step.validation_prompt.format(**entry)
            if step.selection_prompt is not None:
                item["selection_prompt"] = step.selection_prompt.format(**entry)
            results.append(item)
        return results
