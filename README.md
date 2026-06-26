# bKash Ticket Sorter API

A customer support ticket classification API built for the SUST CSE Carnival 2026 Hackathon.

## What It Does
Receives a customer complaint message and classifies it by type, severity, department, and generates an agent summary.

## Tech Stack
- FastAPI
- Gemini 2.5 Flash lite (Google AI)
- Python 3.11
- Deployed on Render

## Setup & Run Locally

1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Create .env file:
   GEMINI_API_KEY=your_key_here
4. Run:
   uvicorn app:app --reload
5. Test at http://localhost:8000/docs

## API Endpoints

GET  /health       → Returns {"status": "ok"}
POST /sort-ticket  → Classifies a customer ticket

## Sample Request
{
  "ticket_id": "T-001",
  "channel": "app",
  "locale": "en",
  "message": "I sent 5000 taka to a wrong number"
}

## Sample Response
{
  "ticket_id": "T-001",
  "case_type": "wrong_transfer",
  "severity": "high",
  "department": "dispute_resolution",
  "agent_summary": "Customer reports sending funds to wrong recipient and requests recovery assistance.",
  "human_review_required": true,
  "confidence": 0.97
}

## AI Usage
Uses Google Gemini 2.5 Flash via API for natural language understanding and ticket classification.
Rule-based logic handles routing and safety checks deterministically.

## Safety Logic
- agent_summary never requests PIN, OTP, password, or card number
- human_review_required is calculated by code, not AI
- Set to true for critical severity or phishing cases
- Gemini fallback returns controlled error if AI service is unavailable

## Known Limitations
- Relies on Gemini API availability
- Free tier on Render may have cold start delays
- Very short or meaningless messages may return low confidence results