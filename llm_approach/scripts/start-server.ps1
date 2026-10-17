# Starts the LLM-approach API on port 8001. Assumes the venv is active and the
# current directory is anything: the script moves to the repo root itself.
#
# --no-access-log is REQUIRED: the WebSocket token is in the URL query string
# (/ws/chat?token=...) and uvicorn's access log would write it to the console.
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
python -m uvicorn llm_approach.server_app:app --port 8001 --no-access-log
