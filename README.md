# AdGPT — Free LLM Tokens with Native Ad Integration

Ad-supported LLM API. Users get free inference; advertisers bid for contextual placement inside responses.

Implements the generative auction from [Zhao et al., "LLM-Auction: Generative Auction towards LLM-Native Advertising" (2025)](https://arxiv.org/abs/2512.10551). The full candidate ad set with bids is passed to the model. The LLM selects the most relevant ad and integrates it natively into its response in a single inference call. A first-price payment rule applies.

## Run

```bash
pip install fastapi uvicorn requests
export GROQ_API_KEY=your_key
python adgpt.py
```

Server starts at `http://localhost:1234`.

## API

OpenAI-compatible. Drop-in replacement for any chat endpoint:

```bash
curl -X POST http://localhost:1234/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Plan a trip to Hawaii"}]}'
```

Response includes the LLM reply with a natively integrated ad, plus an `auction` object showing which advertiser won and their payment.

## Stack

- **Model:** `llama-3.1-8b-instant` via Groq
- **Framework:** FastAPI + Uvicorn
- **Paper:** [LLM-Auction (arXiv:2512.10551)](https://arxiv.org/abs/2512.10551)
