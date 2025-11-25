from __future__ import annotations

from typing import Any
from datetime import datetime, timedelta
import os

from fastapi import APIRouter, HTTPException, Body, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# --- NEW IMPORTS FOR INTELLIGENCE ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    # Fallback imports in case you are using an older version of LangChain
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
# ------------------------------------

from app.adapters.registry import AdapterRegistry
from app.types import (
    NormalizedEvent,
    SendMessageRequest,
    SendMessageResponse,
    IMessageTextMessage,
)
from app.agents.langchain_agent import generate_reply_with_langchain
from app.services.supabase_rag import get_rag_service
from app.services.user_service import get_user_service
from app.services.submission_service import get_submission_service


router = APIRouter(prefix="", tags=["messaging"])
security = HTTPBearer(auto_error=False)

# Initialize services (singletons)
_rag_service = get_rag_service()
_user_service = get_user_service()
_submission_service = get_submission_service()

# Message deduplication: track processed messages to prevent duplicate processing
# Using a simple in-memory cache with TTL (cleans up after 1 hour)
# Key format: "message_id|recipient|text_hash" for better deduplication
_processed_messages: dict[str, datetime] = {}
_MESSAGE_CACHE_TTL = timedelta(hours=1)
_RATE_LIMIT_WINDOW = timedelta(seconds=10)  # Don't process same user messages within 10 seconds


def _cleanup_old_messages() -> None:
    """Remove messages older than TTL from the cache."""
    now = datetime.now()
    expired = [key for key, timestamp in _processed_messages.items() if now - timestamp > _MESSAGE_CACHE_TTL]
    for key in expired:
        del _processed_messages[key]


def _get_message_key(message_id: str | None, recipient: str | None, text: str) -> str:
    """Generate a unique key for message deduplication."""
    import hashlib
    text_hash = hashlib.md5(text.strip().lower().encode()).hexdigest()[:8]
    return f"{message_id or 'no-id'}|{recipient or 'no-recipient'}|{text_hash}"


def _is_message_processed(message_id: str | None, recipient: str | None, text: str) -> bool:
    """Check if a message has already been processed recently."""
    _cleanup_old_messages()
    key = _get_message_key(message_id, recipient, text)
    
    if key in _processed_messages:
        # Check if it's within rate limit window
        last_processed = _processed_messages[key]
        if datetime.now() - last_processed < _RATE_LIMIT_WINDOW:
            return True
    
    return False


def _mark_message_processed(message_id: str | None, recipient: str | None, text: str) -> None:
    """Mark a message as processed."""
    key = _get_message_key(message_id, recipient, text)
    _processed_messages[key] = datetime.now()


# --- NEW INTELLIGENT PROCESSING FUNCTION ---
async def _process_input_intelligently(raw_text: str) -> tuple[str, str]:
    """
    1. Corrects spelling/grammar of the input using a fast LLM.
    2. Retrieves RAG context based on the CORRECTED text.
    3. Appends a 'General Q&A' instruction to the context to allow off-topic answers.
    
    Returns: (corrected_text, enhanced_context)
    """
    corrected_text = raw_text
    
    # 1. Spelling Correction
    try:
        # Using gpt-3.5-turbo or similar fast model for quick correction
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        messages = [
            SystemMessage(content=(
                "You are a text cleaner. Your task is to correct any spelling or grammatical errors "
                "in the user's message to make it suitable for a search engine. "
                "Do not change the meaning. If the text is already correct or is a casual greeting, return it exactly as is. "
                "Output ONLY the corrected text."
            )),
            HumanMessage(content=raw_text)
        ]
        # Await the async call to avoid blocking
        result = await llm.ainvoke(messages)
        corrected_text = result.content.strip()
        
        # Safety check: if LLM returns empty, revert to raw
        if not corrected_text:
            corrected_text = raw_text
            
    except Exception as e:
        print(f"Warning: Spelling correction failed ({e}). Using raw text.")
        corrected_text = raw_text

    # 2. Fetch Context (using corrected text for better matching)
    try:
        context = _rag_service.get_context_for_query(corrected_text)
    except Exception as e:
        print(f"Warning: RAG service failed ({e}).")
        context = ""

    # 3. General Q&A Fallback Instruction
    # We append this to the context so the downstream agent knows it's okay to answer generally.
    general_instruction = (
        "\n\n[SYSTEM INSTRUCTION: The user's query might be about general topics (weather, small talk, general knowledge) "
        "that are not in the database. If the retrieved context above is empty or irrelevant to the user's specific question, "
        "please ignore the context and answer the user's question helpfully using your general knowledge. "
        "Do not say 'I don't know' just because it's not in the context. Be conversational.]"
    )
    
    enhanced_context = (context or "") + general_instruction
    
    return corrected_text, enhanced_context
# -------------------------------------------


@router.post("/messages/send")
async def send_message(payload: SendMessageRequest) -> SendMessageResponse:
    adapter = AdapterRegistry.get(payload.provider)
    try:
        payload.message.ensure_valid_target()
        result = adapter.send_message(payload.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Send error: {e}")
    return SendMessageResponse(ok=True, result=result)


@router.post("/webhooks/{provider}")
async def webhook_events(
    provider: str,
    payload: dict[str, Any] = Body(..., description="Raw webhook JSON payload"),
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> dict[str, Any]:
    """Generic webhook handler for any provider.

    - Verifies the request using the adapter's `verify_request`
    - Normalizes inbound payload to `NormalizedEvent`
    - Calls the agent to generate replies
    - If a recipient is present, sends a text reply via the same adapter
    """
    print(f"WEBHOOK RECEIVED - Provider: {provider}")
    
    adapter = AdapterRegistry.get(provider)

    try:
        token = credentials.credentials if credentials is not None else None
        adapter.verify_request(token)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Unauthorized webhook")

    normalized: NormalizedEvent = adapter.normalize_event(
        payload if isinstance(payload, dict) else {}
    )

    alert_type = normalized.alert_type
    message_id = normalized.message_id
    recipient = normalized.recipient
    text = normalized.text or ""
    
    if alert_type and alert_type != "message_inbound":
        print(f"Ignoring non-message event: alert_type={alert_type}")
        return {"ok": True, "ignored": True, "reason": f"Not a user message (alert_type: {alert_type})"}

    import os
    bot_sender = os.environ.get("LOOP_SENDER_NAME", "chip@ai.imsg.bot")
    if recipient and recipient.lower() == bot_sender.lower():
        print(f"Ignoring message from bot itself: recipient={recipient}")
        return {"ok": True, "ignored": True, "reason": "Message from bot itself"}

    is_duplicate = _is_message_processed(message_id, recipient, text)
    if is_duplicate:
        return {"ok": True, "ignored": True, "reason": "Message already processed recently"}

    user_id = None
    if recipient:
        user_id = _user_service.get_or_create_user(phone_number=recipient)

    if recipient and text and text.strip():
        try:
            # --- MODIFIED LOGIC START ---
            # Instead of getting context directly, we process the input first
            # to fix spelling and prepare the context for general queries.
            corrected_text, enhanced_context = await _process_input_intelligently(text)
            
            # Debug log to see what changed
            print(f"Original: {text} | Corrected: {corrected_text}")

            reply_text, submission_data = generate_reply_with_langchain(
                user_message=corrected_text,  # Pass the fixed text
                context=enhanced_context,     # Pass the enhanced context
                user_id=str(user_id) if user_id else None,
            )
            # --- MODIFIED LOGIC END ---
            
            if submission_data and user_id:
                try:
                    from uuid import UUID
                    submission = _submission_service.create_submission(
                        user_id=UUID(str(user_id)),
                        challenge_id=UUID(submission_data["challenge_id"]),
                        submission_text=submission_data["submission_text"],
                        submission_url=submission_data.get("submission_url"),
                    )
                    if submission and "submission" not in reply_text.lower():
                        reply_text = f"Thanks! I've recorded your submission.\n\n{reply_text}"
                except Exception as e:
                    print(f"Error creating submission: {e}")
            
            _mark_message_processed(message_id, recipient, text)
            
            try:
                adapter.send_message(
                    IMessageTextMessage(
                        recipient=recipient,
                        text=reply_text,
                    )
                )
            except Exception as send_error:
                error_str = str(send_error)
                if "280" not in error_str and "opted out" not in error_str.lower():
                    print(f"Error sending message: {send_error}")
        except Exception as e:
            _mark_message_processed(message_id, recipient, text)
            import traceback
            print(f"ERROR: LangChain response generation failed: {e}")
            print(traceback.format_exc())
            return {"ok": True, "error": "Failed to generate reply", "message_processed": True}
    elif recipient:
        _mark_message_processed(message_id, recipient, text if text else "")
        try:
            adapter.send_message(
                IMessageTextMessage(
                    recipient=recipient,
                    text="Thanks for your message! I'm Chip, your Alabama tech community AI agent. How can I help you today?",
                )
            )
        except Exception as e:
            if "280" not in str(e) and "opted out" not in str(e).lower():
                print(f"Error sending acknowledgment: {e}")

    return {
        "ok": True,
    }
