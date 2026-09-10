# Homilivo WS Demo Client (Phase 3a)

A minimal, standalone Vite + React + TypeScript app that connects to the
Homilivo chatbot's `/ws/chat` WebSocket endpoint. It exists to prove the
endpoint works end-to-end and to serve as a reference client — it is
**not** the production Homilivo UI (that's Phase 3b).

**Source of truth for the protocol:** [`../docs/WEBSOCKET_CONTRACT.md`](../docs/WEBSOCKET_CONTRACT.md).
Everything this app does — connect URL, auth, event shapes, heartbeat,
close codes, reconnect behavior, REST fallback — follows that document.
If you need to change how this client talks to the backend, read the
contract first.

This app does not modify, and does not need to modify, any backend code
under `app/` or `tests/`.

## Install and run

```bash
cd frontend-demo
npm install
cp .env.example .env   # then edit .env, see below
npm run dev
```

Open the URL Vite prints (typically http://localhost:5173).

## Environment variables

Set these in `frontend-demo/.env` (never commit this file — see
`.env.example` for placeholders):

| Variable            | Meaning                                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| `VITE_BACKEND_URL`  | Base HTTP(S) URL of the running backend, e.g. `http://127.0.0.1:8000`. The WebSocket URL is derived from this (`http`→`ws`, `https`→`wss`). |
| `VITE_DEMO_TOKEN`   | Token sent as `?token=<token>` on the WebSocket connect and as `Authorization: Bearer <token>` on the REST fallback. Use a short-lived demo token (`GET /chat-ui/demo-token` on the backend, if a server-side credential is configured there) rather than a real Homilivo API token. |

The backend must already be running (`python -m uvicorn app.api:app --port 8000` from the repo root) with the corresponding port.

## What it does

- `src/useChatSocket.ts` — the hook implementing the contract: opens the
  WebSocket with the token as a query param, sends one turn at a time
  (`{message, session_id, mode}`), handles every event
  (`typing`/`token`/`done`/`error`/`ping`), tracks `session_id` across
  turns, reconnects lazily on the next message after a drop (resending
  the last `session_id` to resume the same conversation), and falls back
  to `POST /chat` whenever the socket can't connect or drops mid-turn.
- `src/ChatWindow.tsx` — message list, text input, and a connection-status
  indicator (`live` / `connecting` / `reconnecting` / `REST fallback`).
- `src/types.ts` — TypeScript types mirroring the contract's event and
  payload shapes exactly.

## Tests

```bash
npm test
```

`src/ChatWindow.test.tsx` renders the chat window, sends a message, and
asserts a streamed reply renders — using a fake `WebSocket` (see
`src/test/fakeWebSocket.ts`) so no live backend is needed.
