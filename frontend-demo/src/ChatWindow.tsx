import { useState } from "react";
import type { FormEvent } from "react";
import { useChatSocket } from "./useChatSocket";
import type { ConnectionStatus, Mode } from "./types";

interface ChatWindowProps {
  backendUrl: string;
  token: string;
}

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  idle: "not connected",
  connecting: "connecting...",
  live: "live",
  reconnecting: "reconnecting...",
  "rest-fallback": "REST fallback",
};

export function ChatWindow({ backendUrl, token }: ChatWindowProps) {
  const { messages, status, isTyping, isSending, sendMessage } =
    useChatSocket({ backendUrl, token });
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("fast");

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || isSending) return;
    setInput("");
    void sendMessage(text, mode);
  };

  return (
    <div className="chat-window">
      <header className="chat-header">
        <h1>Homilivo Demo Client</h1>
        <div className="chat-header-controls">
          <span
            className={`status-badge status-${status}`}
            data-testid="connection-status"
          >
            {STATUS_LABEL[status]}
          </span>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            aria-label="Response mode"
          >
            <option value="fast">fast</option>
            <option value="qwen">qwen</option>
            <option value="gemma">gemma</option>
            <option value="groq">groq</option>
          </select>
        </div>
      </header>

      <div className="message-list" data-testid="message-list">
        {messages.map((m) => (
          <div key={m.id} className={`message-row message-row-${m.role}`}>
            <div
              className={`message-bubble${m.isError ? " message-error" : ""}`}
            >
              {m.text}
            </div>
            {m.rooms && m.rooms.length > 0 && (
              <ul className="room-list">
                {m.rooms.map((room) => (
                  <li key={room.id} className="room-card">
                    <div className="room-title">{room.title}</div>
                    <div className="room-meta">
                      {room.location}, {room.city} · {room.room_type} · ₹
                      {room.price_per_month}/month
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
        {isTyping && (
          <div className="message-row message-row-assistant">
            <div className="message-bubble message-typing">
              Assistant is typing...
            </div>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          maxLength={500}
          placeholder="e.g. Find rooms in Jaipur"
          onChange={(e) => setInput(e.target.value)}
          disabled={isSending}
          aria-label="Message"
        />
        <button type="submit" disabled={isSending || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
