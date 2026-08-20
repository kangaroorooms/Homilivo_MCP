from fastapi import FastAPI
from pydantic import BaseModel, Field

from mcp import Client
from app.server import mcp

from app.chatbot import process_chat


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Homilivo Room Search API",
    description="FastAPI service for the Homilivo MCP AI agent",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Room-search request from the user"
    )


# ============================================================
# ROOM RESPONSE MODEL
# ============================================================

class RoomResponse(BaseModel):

    id: str
    title: str
    city: str

    location: str | None = None

    price_per_month: float | None = None

    room_type: str | None = None

    ideal_for: list[str] = []


# ============================================================
# CHAT RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):

    success: bool

    message: str

    room_count: int = 0

    rooms: list[RoomResponse] = []


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "success",
        "message": "Homilivo Room Search API is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat_endpoint(
    request: ChatRequest
):

    async with Client(mcp) as client:

        result = await process_chat(
            request.message,
            client
        )

    return result