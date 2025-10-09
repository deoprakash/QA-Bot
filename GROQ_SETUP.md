# GROQ API Setup Instructions

## Getting Your Groq API Key

1. Go to https://console.groq.com/
2. Sign up or log in to your account
3. Navigate to the API Keys section
4. Create a new API key
5. Copy the API key

## Add to your .env file

Add this line to your `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Available Models on Groq

| Model | Description | Use Case |
|-------|-------------|----------|
| `llama-3.1-8b-instant` | Fastest model | Quick responses, simple tasks |
| `llama-3.1-70b-versatile` | Balanced model | General purpose, good quality |
| `mixtral-8x7b-32768` | Large context | Long documents, complex reasoning |
| `gemma2-9b-it` | Instruction tuned | Following specific instructions |

## Why Use Groq?

- **Speed**: Up to 10x faster inference than traditional cloud providers
- **Cost-effective**: Competitive pricing
- **Quality**: Access to top open-source models
- **Reliability**: High uptime and consistent performance

## Installation

```bash
pip install langchain-groq
```

## Usage Example

```python
from rag import query_data

# Your .env should contain GROQ_API_KEY
answer = query_data("What is Natural Language Processing?")
print(answer)
```