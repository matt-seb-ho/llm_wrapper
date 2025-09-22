from dataclasses import dataclass

from openai.types import CompletionUsage


@dataclass
class ModelTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    completions: int = 0

    def update(
        self, completion_usage: CompletionUsage, num_completions: int = 1
    ) -> None:
        self.input_tokens += completion_usage.prompt_tokens
        self.output_tokens += completion_usage.completion_tokens

        # record reasoning tokens if available
        added_explicit_reasoning_tokens = False
        if (
            completion_usage.completion_tokens_details
            and completion_usage.completion_tokens_details.reasoning_tokens
        ):
            self.reasoning_tokens += (
                completion_usage.completion_tokens_details.reasoning_tokens
            )
            added_explicit_reasoning_tokens = True

        # ensure reasoning tokens are accounted for in output tokens
        prompt_completion_sum = (
            completion_usage.prompt_tokens + completion_usage.completion_tokens
        )
        if prompt_completion_sum < completion_usage.total_tokens:
            # implies reasoning tokens were used but not reported as part of output tokens
            # - we need to infer from: prompt + completion + reasoning = total
            # - OAI API includes reasoning as part of completions tokens
            # - some other providers (e.g. XAI) do not, so we should add it to be consistent
            inferred_reasoning = completion_usage.total_tokens - prompt_completion_sum
            self.output_tokens += inferred_reasoning
            if not added_explicit_reasoning_tokens:
                self.reasoning_tokens += inferred_reasoning

        self.requests += 1
        self.completions += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "completions": self.completions,
        }
