# Homilivo Chatbot WebSocket Contract — v1

This is the contract for `/ws/chat`. It is everything a frontend
developer needs to integrate against the endpoint **without reading
server code** — the basic demo React client (Phase 3a) and Homilivo's
real frontend (Phase 3b) both build against this document, not against
`app/api.py` directly.

**Versioning:** this is v1. The event shapes, close codes, and auth
mechanism below are a contract — they will not change without a version
bump (v2) and advance notice, per the project's guardrails. What *can*
change without a version bump: how a "done" payload's message is
produced internally (e.g. Phase 8 will make token streaming genuinely
token-by-token instead of today's word-chunked full response) — the
wire-level event shapes stay the same either way.

---

## 1. Connecting

```
ws://<host>/ws/chat?token=<token>       (plain HTTP deployments)
wss://<host>/ws/chat?token=<token>      (HTTPS deployments — always use wss in production)
```

- The token travels as a **query parameter**, not an `Authorization`
  header — browsers' WebSocket API cannot set custom headers on the
  handshake.
- `token` is either:
  - a real Homilivo API token (the same "bring your own token" value
    you'd send as `Authorization: Bearer <token>` to REST `/chat`), or
  - an opaque, short-lived **demo token** obtained from
    `GET /chat-ui/demo-token` (only available when the server operator
    has configured a server-side credential; 404s otherwise). A demo
    token is redeemed server-side for the real credential — it is
    never the real secret itself.
- **Missing token:** the connection is closed immediately (before
  `accept()`) with close code **4401**, reason `"Missing token"`.
- **No token format is validated at connect time.** Any non-empty
  string is provisionally accepted — this mirrors REST `/chat`, which
  also does not validate the token's value, only that one was
  supplied. An actually-wrong token is only discovered when the server
  calls the Homilivo API on your behalf, and surfaces as a normal
  `{"event": "error"}` for that turn (see §4), not a close.
- **Rate limiting:** the connection attempt itself is checked against
  the same per-peer-IP budget REST `/chat` uses (`app/rate_limit.py`).
  If that budget is already exhausted, the connection is closed
  immediately (before `accept()`) with close code **4429**, reason
  `"Rate limit exceeded"`. Note that a *successful* connection attempt
  itself consumes one unit of that same budget — a client that
  reconnects in a tight loop competes with its own message quota.

---

## 2. Sending a turn

Once connected, send **one JSON object per turn**:

```json
{
  "message": "find rooms in Jaipur",
  "session_id": null,
  "mode": "fast"
}
```

| field        | type                                      | required | notes                                                                 |
|--------------|-------------------------------------------|----------|------------------------------------------------------------------------|
| `message`    | string, 1–500 characters                  | yes      | Empty, whitespace-only, or over 500 chars → `{"event":"error"}`.     |
| `session_id` | string \| null                            | no       | Omit or `null` on the first turn. See §5 for resuming a session.     |
| `mode`       | `"fast"` \| `"qwen"` \| `"gemma"` \| `"groq"` | no    | Defaults to `"fast"`. Any other value → `{"event":"error"}`.         |

You may send multiple turns over one connection, one at a time (wait
for that turn's `"done"` or `"error"` before sending the next).

---

## 3. Event sequence for a normal turn

The server replies with a sequence of JSON text frames for that turn,
in this exact order:

```json
{"event": "typing"}
```
```json
{"event": "token", "text": "Found "}
{"event": "token", "text": "3 rooms "}
{"event": "token", "text": "in Jaipur..."}
```
(repeated, zero or more times, in order — concatenating every `text`
in the order received reconstructs the full `message` in the `"done"`
event below)
```json
{
  "event": "done",
  "success": true,
  "session_id": "b3b2e2b0-...-9c1e",
  "request_id": "0f2a7e6c-...-4d1b",
  "mode": "fast",
  "message": "Found 3 rooms in Jaipur...",
  "room_count": 3,
  "rooms": [
    {
      "id": "room-1",
      "title": "Sunrise 2BHK",
      "city": "Jaipur",
      "location": "Jagatpura",
      "price_per_month": 18000,
      "room_type": "2BHK",
      "ideal_for": ["Family", "Working Professionals"],
      "amenities": ["AC", "Parking"]
    }
  ],
  "selected_room": null,
  "pending_clarification": false,
  "sources": [],
  "search_performed": true
}
```

`"done"`'s fields (everything except the `"event"` key) are **exactly**
the REST `/chat` response body's fields — see §6.

`request_id` identifies this turn for `POST /feedback` (thumbs up/down).

---

## 4. Errors (do not close the connection)

A problem with *one specific turn* — as opposed to the connection
itself — is reported as an error event, and the connection stays open
and usable for the next turn:

```json
{"event": "error", "detail": "message must be a non-empty string of at most 500 characters."}
```

This is sent for:
- a message that is not valid JSON, or not a JSON object,
- `message` missing, empty, or over 500 characters,
- an invalid `mode`,
- the per-peer-IP rate limit being exceeded on this message (distinct
  from the connect-time 4429 close — see §1; the connection is *not*
  closed here),
- an unhandled error while processing the turn (the detail is always
  the fixed string `"Something went wrong processing that message.
  Please try again."` — no internal error text or stack trace is ever
  sent to the client).

---

## 5. Heartbeat

If the server has not received a message for `WS_HEARTBEAT_INTERVAL_SECONDS`
(default **20 seconds**, configurable server-side) it sends:

```json
{"event": "ping"}
```

This only happens between turns, while the server is idle waiting for
your next message — never in the middle of a turn's typing/token/done
sequence. Clients should ignore this event; its only purpose is to let
you distinguish "connection alive but quiet" from "connection dead" so
you know when to reconnect.

---

## 6. REST fallback (`POST /chat`)

Use REST whenever a WebSocket cannot be opened, drops mid-turn, or you
simply prefer a non-streaming integration. Same request shape, real
`Authorization` header instead of a query param:

```
POST /chat
Authorization: Bearer <token>
Content-Type: application/json

{"message": "find rooms in Jaipur", "session_id": null, "mode": "fast"}
```

The 200 response body is **exactly** the same fields as the WS
`"done"` event, minus the `"event"` key — same keys, same types, same
values for the same query. There is no separate "REST schema" to
track; both transports are built from the same internal turn result.

---

## 7. Close codes

| Code | Meaning                                                    | When                                              |
|------|-------------------------------------------------------------|----------------------------------------------------|
| 4401 | Missing token                                                | On connect, before `accept()`.                     |
| 4429 | Rate limit exceeded                                          | On connect, before `accept()`.                     |
| 1000 / 1001 / etc. | Standard WebSocket closure codes                | Normal client-initiated close, network drop, server shutdown — not app-specific. |

4401 and 4429 are in the [IANA-reserved app range](https://www.rfc-editor.org/rfc/rfc6455) (4000–4999) and are only ever sent **before** the connection is accepted — once a connection is open, per-turn problems are reported as `{"event": "error"}` (§4), not a close.

---

## 8. Reconnecting and session continuity

There is **no automatic server-initiated reconnect** and no in-band
resume handshake. If a connection drops (network issue, server
restart, or the client explicitly closes it):

1. The client is responsible for either opening a new WebSocket
   connection or falling back to REST `/chat` for the next message.
2. **To resume the same conversation**, the client must have persisted
   the `session_id` value it received in the last `"done"` event and
   include that same `session_id` in the first message of the new
   connection (or the REST fallback call). Everything the assistant
   needs to remember — room results, follow-up context, pending
   clarifications — is keyed by `session_id` server-side (see
   `app/session_store.py`), not by the connection.
3. On a brand-new conversation, omit `session_id` (or send `null`) and
   start using whatever value comes back in the first `"done"` event
   from then on.
4. When the server is deployed with `STATE_BACKEND=redis` (see
   `.env.example`), session state is shared across server instances,
   so resuming by `session_id` works even if the reconnect lands on a
   different process than the original connection — not just within
   one process.

The bundled demo page (`app/static/chat_ui.html`) implements this as:
try the WebSocket first; if it cannot connect or drops mid-turn, retry
that turn over REST; reconnect lazily on the next user message rather
than eagerly in the background.

---

## 9. What this version does **not** do

- **Streaming is not yet token-by-token.** Today's `"token"` events
  are the finished response chunked word-by-word after the fact, not
  emitted as the model generates them. Phase 8 will make this
  genuinely incremental — the event shapes above will not change when
  it does.
- **No mid-connection re-authentication.** The token is only checked
  (for presence) at connect time; it is re-resolved (demo token →
  real credential) on every turn, but the connection itself is not
  re-authorized.
