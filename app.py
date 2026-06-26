# pyrefly: ignore [missing-import]
from google import genai
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Optional 
import json
# pyrefly: ignore [missing-import]
from fastapi import HTTPException,Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import logging 
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

limiter = Limiter(key_func=get_remote_address)
app=FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class TicketStr(BaseModel):
    ticket_id:str
    channel:Optional[str]=None
    locale:Optional[str]=None
    message:str

@app.get("/")
async def root():
    return {"message": "Welcome to the Ticket Classifier API"}

@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok"}

@app.post("/sort-ticket")
@limiter.limit("110/minute")
async def sort_ticket(request:Request, ticket:TicketStr):
    start_time=time.time()
    if not ticket.message:
        raise HTTPException(status_code=400, detail="Message is required")
    if not ticket.ticket_id:
        raise HTTPException(status_code=400, detail="Ticket ID is required")
    logger.info(f"Received ticket | id={ticket.ticket_id} | channel={ticket.channel}")
    prompt = f"""
You are a classifier for a digital finance support system in Bangladesh.

Given a customer message, return ONLY a valid JSON object with these exact fields:
- case_type: one of [wrong_transfer, payment_failed, refund_request, phishing_or_social_engineering, other]
- severity: one of [low, medium, high, critical]
- department: one of [customer_support, dispute_resolution, payments_ops, fraud_risk]
- agent_summary: 1-2 neutral sentences. NEVER mention PIN, OTP, password, or card number.
- confidence: a float between 0.0 and 1.0

Routing rules:
- wrong_transfer → dispute_resolution, severity high
- payment_failed → payments_ops, severity high
- phishing_or_social_engineering → fraud_risk, severity critical
- simple refund → customer_support, severity low
- other → customer_support, severity low

Customer message: "{ticket.message}"

Return ONLY the JSON object. No explanation, no markdown, no extra text.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw = response.text.strip()

        # Strip markdown if Gemini wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON from Gemini: {str(e)}")
    human_review = (
    result["severity"] in ["critical", "high"]
    or result["case_type"] == "phishing_or_social_engineering"
)   
    end_time=time.time()
    elapsed=round(end_time-start_time,4)
    logger.info(f"Processed ticket | id={ticket.ticket_id} | time={elapsed}s")

    return {
        "ticket_id": ticket.ticket_id,
        "case_type": result["case_type"],
        "severity": result["severity"],
        "department": result["department"],
        "agent_summary": result["agent_summary"],
        "human_review_required": human_review,
        "confidence": result["confidence"]
    }