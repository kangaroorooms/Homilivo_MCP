import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatWindow } from "./ChatWindow";
import { FakeWebSocket } from "./test/fakeWebSocket";

describe("ChatWindow", () => {
  beforeEach(() => {
    FakeWebSocket.instances.length = 0;
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  });

  it("renders, sends a message, and renders a streamed reply", async () => {
    const user = userEvent.setup();
    render(
      <ChatWindow backendUrl="http://localhost:8000" token="demo-token" />,
    );

    expect(
      screen.getByRole("heading", { name: "Homilivo Demo Client" }),
    ).toBeInTheDocument();

    const input = screen.getByLabelText("Message");
    await user.type(input, "Find rooms in Jaipur");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Find rooms in Jaipur")).toBeInTheDocument();

    // The hook opens a WebSocket per docs/WEBSOCKET_CONTRACT.md §1
    // (token as a query param) before sending the turn.
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const ws = FakeWebSocket.instances[0];
    expect(ws.url).toBe("ws://localhost:8000/ws/chat?token=demo-token");

    ws.simulateOpen();

    await waitFor(() => expect(ws.sent).toHaveLength(1));
    expect(JSON.parse(ws.sent[0])).toEqual({
      message: "Find rooms in Jaipur",
      session_id: null,
      mode: "fast",
    });

    ws.simulateMessage({ event: "typing" });
    expect(await screen.findByText("Assistant is typing...")).toBeInTheDocument();

    ws.simulateMessage({ event: "token", text: "Found " });
    ws.simulateMessage({ event: "token", text: "3 rooms in Jaipur." });
    ws.simulateMessage({
      event: "done",
      success: true,
      session_id: "session-1",
      request_id: "req-1",
      mode: "fast",
      message: "Found 3 rooms in Jaipur.",
      room_count: 0,
      rooms: [],
      selected_room: null,
      pending_clarification: false,
      sources: [],
      search_performed: true,
    });

    expect(
      await screen.findByText("Found 3 rooms in Jaipur."),
    ).toBeInTheDocument();
    expect(screen.getByTestId("connection-status")).toHaveTextContent("live");
  });
});
