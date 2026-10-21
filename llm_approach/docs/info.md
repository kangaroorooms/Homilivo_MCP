Window 1: backend, from C:\Faizal\AI intern\Homilivo chatbot:

powershell

.\venv\Scripts\Activate.ps1

Invoke-WebRequest https://homilivo.onrender.com/api/rooms -UseBasicParsing -TimeoutSec 90 | Out-Null

.\llm_approach\scripts\start-server.ps1

Window 2: frontend, from C:\Faizal\AI intern:


cd "C:\Faizal\AI intern"

.\start-frontend-llm.ps1


Gemini free-tier limits (gemini-3.5-flash-lite): 15 requests/min, 250K tokens/min, 500 requests/day (binding). No daily token limit from Google; LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET is our own safety cap.

Add to .env (repo root):
LLM_APPROACH_GEMINI_SPACING_S=5
LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET=2000000
(optional) LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET=450  (default; the day ends at midnight US Pacific time)

From C:\Faizal\AI intern\Homilivo chatbot:
python -m llm_approach.llm_info                        # provider, model, requests used today, budgets
python -m llm_approach.eval.plan                       # requests/minutes per dataset, order per day
python -m llm_approach.eval.run_eval --dataset golden  # prints its plan, refuses if it does not fit today
python -m llm_approach.eval.run_eval --dataset golden --resume   # after "STOPPED EARLY": use the exact command it printed
