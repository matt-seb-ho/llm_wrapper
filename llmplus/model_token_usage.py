from dataclasses import dataclass

from openai.types import CompletionUsage


@dataclass
class ModelTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    completions: int = 0

    def update(
        self, completion_usage: CompletionUsage, num_completions: int = 1
    ) -> None:
        self.input_tokens += completion_usage.prompt_tokens
        self.output_tokens += completion_usage.completion_tokens
        self.requests += 1
        self.completions += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "completions": self.completions,
        }
