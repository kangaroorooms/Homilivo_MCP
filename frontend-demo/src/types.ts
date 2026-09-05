// Types mirror docs/WEBSOCKET_CONTRACT.md exactly. Do not add fields that
// aren't in the contract.

export type Mode = "fast" | "qwen" | "gemma" | "groq";

export interface Room {
  id: string;
  title: string;
  city: string;
  location: string;
  price_per_month: number;
  room_type: string;
  ideal_for: string[];
  amenities: string[];
}

// The "done" event's fields (minus "event") are exactly the REST /chat
// response body's fields (contract §3, §6).
export interface TurnResult {
  success: boolean;
  session_id: string;
  request_id: string;
  mode: Mode;
  message: string;
  room_count: number;
  rooms: Room[];
  selected_room: Room | null;
  pending_clarification: boolean;
  sources: string[];
  search_performed: boolean;
}

export type ServerEvent =
  | { event: "typing" }
  | { event: "token"; text: string }
  | ({ event: "done" } & TurnResult)
  | { event: "error"; detail: string }
  | { event: "ping" };

export interface OutgoingTurn {
  message: string;
  session_id: string | null;
  mode: Mode;
}

export type ConnectionStatus =
  | "idle"
  | "connecting"
  | "live"
  | "reconnecting"
  | "rest-fallback";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  isError?: boolean;
  rooms?: Room[];
}
