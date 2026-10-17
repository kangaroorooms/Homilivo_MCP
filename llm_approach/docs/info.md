Window 1: backend, from C:\Faizal\AI intern\Homilivo chatbot:

powershell

.\venv\Scripts\Activate.ps1

Invoke-WebRequest https://homilivo.onrender.com/api/rooms -UseBasicParsing -TimeoutSec 90 | Out-Null

.\llm_approach\scripts\start-server.ps1

Window 2: frontend, from C:\Faizal\AI intern:


cd "C:\Faizal\AI intern"

.\start-frontend-llm.ps1