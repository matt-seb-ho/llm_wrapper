# llmplus
A wrapper around openai-python's `AsyncOpenAI` client that provides some extra functionality:
- caching responses (saved to a diskcache)
- retrying
- batching operation
