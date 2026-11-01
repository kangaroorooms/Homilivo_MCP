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
(optional) LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET=450  (default; counted over a rolling 24 h, not a calendar day)
(optional) LLM_APPROACH_GEMINI_DAY_RESET=07:00           (UTC; fixed daily boundary, off by default - only once Google's real boundary is known)
Note: Google's reset boundary was NOT midnight Pacific as first assumed (our counter reset, AI Studio's did not). Limits are now a rolling 24 h window for requests and tokens.

From C:\Faizal\AI intern\Homilivo chatbot:
python -m llm_approach.llm_info                        # provider, model, requests/tokens used in the last 24 h, budgets, when capacity frees up
python -m llm_approach.llm_info --seed-used 204 --seed-tokens 150000 --seed-at 12:36   # copy the number from Google's dashboard (max 500)
python -m llm_approach.eval.plan                       # requests/minutes per dataset, order per day
python -m llm_approach.eval.run_eval --dataset golden  # prints its plan, refuses if it does not fit in the last 24 h
python -m llm_approach.eval.run_eval --dataset golden --resume   # after "STOPPED EARLY": use the exact command it printed
