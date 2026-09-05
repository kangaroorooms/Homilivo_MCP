# python -m uvicorn app.api:app --reload --port 8000
# http://127.0.0.1:8000/docs#/default/chat_endpoint_chat_post
# curl http://127.0.0.1:8000/
# $env:KANGAROO_TOKEN="YOUR_ACTUAL_TOKEN"
#  .\venv\Scripts\Activate.ps1
# python -m app.experiments.agent_modes

# python -m uvicorn app.api:app --port 8000
Fastapi
# http://127.0.0.1:8000/docs
chatui
# http://127.0.0.1:8000/chat-ui

#model=GPT-OSS 20B
#all-MiniLM-L6-v2 embedding model

#The current implementation keeps this in memory, with:
1-hour session TTL
Maximum 1,000 sessions
Maximum 40 messages per session.

#A rate limiter was added:
30 requests / 60 seconds / IP

#python -m pytest -v

#python tests/run_golden_eval.py