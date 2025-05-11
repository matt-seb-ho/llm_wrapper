# llmplus
A wrapper around openai-python's `AsyncOpenAI` client that provides some extra functionality:
- caching responses (saved to a diskcache)
- retrying
- batching operation

## installation
1. clone the repo
2. `cd` into repo
3. pip install -e .
