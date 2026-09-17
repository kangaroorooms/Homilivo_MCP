from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.chatbot.service import chat_with_homilivo


router = APIRouter(
    prefix="/api/chat",
    tags=["Chatbot"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    previous_interaction_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    interaction_id: str
    sources: list[dict[str, str]]


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return chat_with_homilivo(
            message=request.message,
            previous_interaction_id=request.previous_interaction_id,
        )

    except Exception as error:
        print(f"Chatbot error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process chatbot request",
        )


# Backend should recognize the things and quota rate limiting things 