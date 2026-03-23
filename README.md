# AdGPT - Free LLM Tokens with Native Ad Integration

Ad-supported LLM API. Users get free inference; advertisers bid for contextual placement inside responses.

Implements the generative auction from [Zhao et al., "LLM-Auction: Generative Auction towards LLM-Native Advertising" (2025)](https://arxiv.org/abs/2512.10551). The full candidate ad set with bids is passed to the model. The LLM selects the most relevant ad and integrates it natively into its response in a single inference call. A first-price payment rule applies.

## Setup

```bash
pip install fastapi uvicorn requests
```

Set your Groq API key (one of):
```bash
export GROQ_API_KEY=your_key
# or
echo "your_key" > ~/.hermes/groq_api_key.txt
```

## Run

```bash
python adgpt.py
```

Starts on `http://localhost:1234`. The web UI has two tabs: **Chat** (demo the ad integration) and **Advertisers** (view the current ad pool and bids).

## API

OpenAI-compatible chat endpoint. Drop-in replacement:

```bash
curl -X POST http://localhost:1234/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Plan a trip to Hawaii"}]}'
```

Response:
```json
{
  "role": "assistant",
  "content": "Hawaii is a wonderful destination... [Ad — Beta Resort] ...",
  "auction": {
    "winner": "Beta Resort",
    "genre": "travel",
    "bid": 65,
    "payment": 65,
    "method": "LLM-Auction first-price"
  }
}
```

The `auction` object shows which advertiser won and their payment. The ad is woven naturally into the response text.

## Stack

- **Model:** `llama-3.1-8b-instant` via [Groq](https://groq.com)
- **Framework:** FastAPI + Uvicorn
- **Paper:** [LLM-Auction (arXiv:2512.10551)](https://arxiv.org/abs/2512.10551)
