// Minimal fake WebSocket for tests — implements just enough of the
// browser WebSocket API for useChatSocket.ts (addEventListener for
// open/error/close/message, send, close, readyState).

type Listener = (event: { data?: string }) => void;

export class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  static instances: FakeWebSocket[] = [];

  url: string;
  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];
  private listeners: Record<string, Listener[]> = {};

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, cb: Listener) {
    (this.listeners[type] ??= []).push(cb);
  }

  send(data: string) {
    if (this.readyState !== FakeWebSocket.OPEN) {
      throw new Error("FakeWebSocket is not open");
    }
    this.sent.push(data);
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
    this.emit("close", {});
  }

  private emit(type: string, event: { data?: string }) {
    for (const cb of this.listeners[type] ?? []) cb(event);
  }

  simulateOpen() {
    this.readyState = FakeWebSocket.OPEN;
    this.emit("open", {});
  }

  simulateClose() {
    this.readyState = FakeWebSocket.CLOSED;
    this.emit("close", {});
  }

  simulateMessage(data: unknown) {
    this.emit("message", { data: JSON.stringify(data) });
  }
}
