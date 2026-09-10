import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ChatMessage,
  ConnectionStatus,
  Mode,
  Room,
  ServerEvent,
  TurnResult,
} from "./types";

// Everything in this hook follows docs/WEBSOCKET_CONTRACT.md exactly:
// connect URL + auth (§1), turn payload (§2), event sequence (§3), errors
// (§4), heartbeat (§5), REST fallback (§6), reconnect/session resume (§8).

const WS_CONNECT_TIMEOUT_MS = 4000;

let messageIdCounter = 0;
function nextMessageId(): string {
  messageIdCounter += 1;
  return `msg-${Date.now()}-${messageIdCounter}`;
}

function wsUrlFor(backendUrl: string, token: string): string {
  const httpUrl = new URL(backendUrl);
  const wsProtocol = httpUrl.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProtocol}//${httpUrl.host}/ws/chat?token=${encodeURIComponent(token)}`;
}

interface UseChatSocketOptions {
  backendUrl: string;
  token: string;
}

interface UseChatSocketResult {
  messages: ChatMessage[];
  status: ConnectionStatus;
  isTyping: boolean;
  isSending: boolean;
  sendMessage: (text: string, mode?: Mode) => Promise<void>;
}

export function useChatSocket({
  backendUrl,
  token,
}: UseChatSocketOptions): UseChatSocketResult {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const currentAssistantIdRef = useRef<string | null>(null);
  const turnSettleRef = useRef<((handled: boolean) => void) | null>(null);

  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  const appendMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const appendAssistantToken = useCallback((text: string) => {
    setMessages((prev) => {
      const id = currentAssistantIdRef.current;
      if (id === null) return prev;
      return prev.map((m) => (m.id === id ? { ...m, text: m.text + text } : m));
    });
  }, []);

  const finalizeAssistantMessage = useCallback(
    (text: string, rooms?: Room[]) => {
      const id = currentAssistantIdRef.current;
      if (id === null) {
        appendMessage({ id: nextMessageId(), role: "assistant", text, rooms });
        return;
      }
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, text, rooms } : m)),
      );
      currentAssistantIdRef.current = null;
    },
    [appendMessage],
  );

  const reportError = useCallback(
    (detail: string) => {
      const id = currentAssistantIdRef.current;
      if (id !== null) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === id ? { ...m, text: detail, isError: true } : m,
          ),
        );
        currentAssistantIdRef.current = null;
        return;
      }
      appendMessage({
        id: nextMessageId(),
        role: "assistant",
        text: detail,
        isError: true,
      });
    },
    [appendMessage],
  );

  const handleServerEvent = useCallback(
    (raw: string) => {
      let data: ServerEvent;
      try {
        data = JSON.parse(raw);
      } catch {
        return;
      }

      switch (data.event) {
        case "ping":
          // Heartbeat only; nothing to do (contract §5).
          return;
        case "typing":
          setIsTyping(true);
          return;
        case "token":
          if (currentAssistantIdRef.current === null) {
            const id = nextMessageId();
            currentAssistantIdRef.current = id;
            setIsTyping(false);
            appendMessage({ id, role: "assistant", text: "" });
          }
          appendAssistantToken(data.text);
          return;
        case "done":
          setIsTyping(false);
          sessionIdRef.current = data.session_id;
          finalizeAssistantMessage(data.message, data.rooms);
          if (turnSettleRef.current) {
            const settle = turnSettleRef.current;
            turnSettleRef.current = null;
            settle(true);
          }
          return;
        case "error":
          setIsTyping(false);
          reportError(data.detail);
          if (turnSettleRef.current) {
            const settle = turnSettleRef.current;
            turnSettleRef.current = null;
            settle(true);
          }
          return;
      }
    },
    [appendAssistantToken, appendMessage, finalizeAssistantMessage, reportError],
  );

  const openSocket = useCallback(
    (url: string): Promise<WebSocket> => {
      return new Promise((resolve, reject) => {
        let settled = false;
        let ws: WebSocket;
        try {
          ws = new WebSocket(url);
        } catch (err) {
          reject(err instanceof Error ? err : new Error(String(err)));
          return;
        }

        const timeoutId = setTimeout(() => {
          if (settled) return;
          settled = true;
          try {
            ws.close();
          } catch {
            // ignore
          }
          reject(new Error("WebSocket connect timeout"));
        }, WS_CONNECT_TIMEOUT_MS);

        ws.addEventListener("open", () => {
          if (settled) return;
          settled = true;
          clearTimeout(timeoutId);
          resolve(ws);
        });

        ws.addEventListener("error", () => {
          if (settled) return;
          settled = true;
          clearTimeout(timeoutId);
          reject(new Error("WebSocket connection failed"));
        });

        ws.addEventListener("close", () => {
          if (socketRef.current === ws) {
            socketRef.current = null;
            setStatus("rest-fallback");
          }
          // A turn mid-flight when the socket dropped must not hang
          // forever waiting for a "done"/"error" that will never arrive
          // (contract §8: client is responsible for reconnect/fallback).
          if (turnSettleRef.current) {
            const settle = turnSettleRef.current;
            turnSettleRef.current = null;
            settle(false);
          }
        });

        ws.addEventListener("message", (event) => {
          handleServerEvent(event.data as string);
        });
      });
    },
    [handleServerEvent],
  );

  const ensureSocket = useCallback(async (): Promise<WebSocket | null> => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      return socketRef.current;
    }
    setStatus(socketRef.current === null ? "connecting" : "reconnecting");
    try {
      const url = wsUrlFor(backendUrl, token);
      const ws = await openSocket(url);
      socketRef.current = ws;
      setStatus("live");
      return ws;
    } catch {
      socketRef.current = null;
      setStatus("rest-fallback");
      return null;
    }
  }, [backendUrl, openSocket, token]);

  const sendViaSocket = useCallback(
    (ws: WebSocket, text: string, mode: Mode): Promise<boolean> => {
      return new Promise((resolve) => {
        turnSettleRef.current = resolve;
        try {
          ws.send(
            JSON.stringify({
              message: text,
              session_id: sessionIdRef.current,
              mode,
            }),
          );
        } catch {
          turnSettleRef.current = null;
          resolve(false);
        }
      });
    },
    [],
  );

  const sendViaRest = useCallback(
    async (text: string, mode: Mode) => {
      setIsTyping(true);
      try {
        const response = await fetch(`${backendUrl}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            message: text,
            session_id: sessionIdRef.current,
            mode,
          }),
        });

        setIsTyping(false);

        if (!response.ok) {
          let detail = response.statusText;
          try {
            const errorBody = await response.json();
            if (errorBody && errorBody.detail) {
              detail =
                typeof errorBody.detail === "string"
                  ? errorBody.detail
                  : JSON.stringify(errorBody.detail);
            }
          } catch {
            // Non-JSON error body; fall back to statusText already set.
          }
          reportError(`Error: ${detail}`);
          return;
        }

        const data: TurnResult = await response.json();
        sessionIdRef.current = data.session_id;
        finalizeAssistantMessage(data.message, data.rooms);
      } catch {
        setIsTyping(false);
        reportError(
          "Could not reach the server. Please check your connection and try again.",
        );
      }
    },
    [backendUrl, finalizeAssistantMessage, reportError, token],
  );

  const sendMessage = useCallback(
    async (text: string, mode: Mode = "fast") => {
      if (isSending) return;
      setIsSending(true);
      appendMessage({ id: nextMessageId(), role: "user", text });

      try {
        const ws = await ensureSocket();
        if (ws) {
          const handled = await sendViaSocket(ws, text, mode);
          if (!handled) {
            setStatus("rest-fallback");
            await sendViaRest(text, mode);
          }
        } else {
          await sendViaRest(text, mode);
        }
      } finally {
        setIsSending(false);
      }
    },
    [appendMessage, ensureSocket, isSending, sendViaRest, sendViaSocket],
  );

  return { messages, status, isTyping, isSending, sendMessage };
}
