class LLM:
    """Simple interface for language model backends."""

    def __init__(self, process_func):
        self.process_func = process_func

    def process(self, prompt: str) -> str:
        """Return model output for ``prompt`` via the wrapped function."""
        return self.process_func(prompt)
