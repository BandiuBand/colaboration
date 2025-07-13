from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple
import json

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
        step_extract = Step(
            task_prompt=(
                "Extract factual statements from the text and replace them with placeholders. "
                "Respond with JSON {\"text\": replaced_text, \"facts\": [facts]}:\n{prompt}"
            ),
            validation_prompt="Ensure the JSON contains 'text' and 'facts' fields.",
            llm=self.llm,
        )
        return [step_gather, step_extract]

    def _extract_facts(self, text: str) -> Tuple[str, List[str]]:
        """Use LLM to identify facts and replace them with placeholders."""
        raw = self.steps[1].execute({"prompt": text})
        data = json.loads(raw)
        replaced = data.get("text", "")
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            raise ValueError("`facts` field must be a list")
        return replaced, facts

    def _generate_replacements(self, facts: List[str], context: str) -> List[str]:
        """Generate random substitutes for each fact using the LLM."""
        replacements: List[str] = []
        for fact in facts:
            prompt = (
                "Given the context below, replace the fact with a random variant that "
                "still makes sense.\nContext:\n" + context + "\nFact: " + fact
            )
            replacements.append(self.llm.process(prompt).strip())
        return replacements

    def _apply_replacements(self, text: str, replacements: List[str]) -> str:
        result = text
        for i, repl in enumerate(replacements, start=1):
            result = result.replace(f"FACT_{i}", repl)
        return result

    def run(self) -> List[Dict[str, Any]]:
        """Execute the workflow and return the augmented dataset."""
        results: List[Dict[str, Any]] = []
        for entry in self.dataset:
            filled = self.steps[0].execute(entry)
            base_text, facts = self._extract_facts(filled)
            for _ in range(self.num_variations):
                subs = self._generate_replacements(facts, filled)
                new_prompt = self._apply_replacements(base_text, subs)
                results.append({"prompt": new_prompt})
        return results
