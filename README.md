# QueueStorm Investigator

**bKash SUST CSE Carnival 2026 — Codex Community Hackathon**
**Team:** ReadyOrNot
**Round:** Online Preliminary Qualification
**June 2026**

---

## Project Overview

QueueStorm Investigator is a robust AI-powered API service built for the Online Preliminary Round of the bKash SUST CSE Carnival 2026 Hackathon. It serves as an intelligent copilot for digital finance support agents, analyzing customer complaints against transaction history to classify issues, determine evidence consistency, route cases to the correct department, and generate safe multilingual responses — all within a strict safety framework.

---

## Tech Stack

- **Framework:** FastAPI (Python)
- **LLM:** Google Gemini (`gemini-2.5-flash` primary + `gemini-2.0-flash` fallback)
- **Validation:** Pydantic
- **Rate Limiting:** slowapi (110 requests/minute)
- **Environment:** python-dotenv
- **Server:** Uvicorn
- **Deployment:** Render (live) + UptimeRobot (uptime monitoring)

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Google Gemini API Key

### Installation

```bash
git clone https://github.com/koalaagainstthemachinelearning/sust_carnival_elimination.git
cd sust_carnival_elimination

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

`.env.example` contents:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Run Command

**Development:**
```bash
uvicorn app:app --reload
```

**Production (local):**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Live Endpoints

- **Base URL:** `https://sust-carnival-elimination.onrender.com`
- **Health Check:** `GET /health` → `{"status": "ok"}`
- **Analysis:** `POST /analyze-ticket` with JSON body per problem statement schema

---

## AI Approach

The system uses a structured system prompt with an explicit 7-step investigation chain sent to Gemini at temperature 0.1 for deterministic, rule-following behavior:

1. **Understand the complaint** — parse amount, time, recipient, transaction type, detect scam/phishing signals
2. **Scan transaction history** — go through each transaction individually to find `relevant_transaction_id`
3. **Determine `evidence_verdict`** — `consistent`, `inconsistent`, or `insufficient_data`
4. **Classify `case_type`** — complaint text always drives classification, never the transaction status
5. **Route to department** — based on case_type and evidence
6. **Set severity** — phishing always critical; wrong_transfer with consistent evidence is high; etc.
7. **Write text fields** — `agent_summary`, `recommended_next_action`, `customer_reply` in correct language

### Key Investigation Rules Built Into the Prompt

- **PRIOR PATTERN RULE:** Repeated prior transfers to same counterparty → `evidence_verdict = inconsistent` on wrong_transfer claims
- **DUPLICATE PAYMENT RULE:** Two transactions, same amount + counterparty + type within minutes → `relevant_transaction_id` is the second (later) one
- **AMBIGUOUS MATCH RULE:** Multiple equally plausible matches with no distinguishing info → `relevant_transaction_id = null`, `evidence_verdict = insufficient_data`. Never guesses.
- **VAGUE COMPLAINT RULE:** No specific amount, time, or type → `insufficient_data`

### Post-Processing Pipeline

After the model responds:
1. Strip markdown fences from raw response
2. Parse JSON
3. Normalize all enums to lowercase
4. Validate every enum against hardcoded valid sets — bad model output never reaches the judge
5. Apply injection override if detected
6. Recalibrate confidence score

---

## Safety Logic

Safety is enforced at **three independent layers**:

**Layer 1 — Pre-processing (Injection Detection):**
Pattern matching across 10 known prompt injection phrases in the complaint text. If detected, the model is still called but the output is hard-overridden regardless of what the model returns:
- `case_type` → `phishing_or_social_engineering`
- `department` → `fraud_risk`
- `severity` → `critical`
- `human_review_required` → `true`
- `evidence_verdict` → `insufficient_data`

**Layer 2 — System Prompt Guardrails:**
The LLM system prompt contains absolute safety rules:
- Never ask for PIN, OTP, password, or card number
- Never confirm a refund, reversal, or recovery
- Never direct customer to third-party or unofficial channels
- Phishing cases: thank customer for reporting, reassure company never asks for credentials

**Layer 3 — Post-Processing Enum Validation:**
All enum fields validated against hardcoded sets after model response. Invalid values are replaced with safe defaults before any response is returned.

All safety rules from Section 8 of the problem statement are fully respected.

---

## MODELS

| Model | Where it runs | Why chosen |
|---|---|---|
| `gemini-2.5-flash` | Google AI API (primary) | Fast, cost-effective, strong multi-step reasoning, excellent JSON output |
| `gemini-2.0-flash` | Google AI API (fallback) | Automatic fallback if primary times out (25s timeout) or fails |

- **Temperature:** 0.1 — for deterministic, consistent, rule-following outputs
- **Max output tokens:** 4092
- **Timeout:** 25 seconds per model, automatic fallback to next model
- **Expected cost:** Very low — Gemini Flash tier is suitable for high-volume support use

---

## Assumptions

- Transaction history provided is truthful and complete for the purpose of investigation
- All inputs follow the provided schema
- Synthetic data only — no real PII involved
- Bangla and Banglish support handled via Gemini's multilingual capabilities
- `language` field in request drives `customer_reply` language selection

---

## Known Limitations

- Relies on LLM reasoning — rare edge cases in highly ambiguous multi-transaction scenarios may produce suboptimal evidence verdicts
- No persistent memory across tickets — each ticket is investigated independently
- Response time depends on Gemini API latency (target under 30 seconds; fallback adds ~25 seconds on primary failure)
- Bangla generation quality depends on model version
- Injection detection uses pattern matching — novel injection phrasing not in the 10 patterns may not be caught at the pre-processing layer (still caught by system prompt guardrails)

---

## Sample Output

A sample output file `sample_output.json` is included in the repository, generated by running the service against `SUST_Preli_Sample_Cases.json`.

---

## Deployment

- **Live URL:** https://sust-carnival-elimination.onrender.com
- **GitHub:** https://github.com/koalaagainstthemachinelearning/sust_carnival_elimination
- **Uptime Monitoring:** UptimeRobot pings `/health` every 5 minutes to keep the Render service alive
- Platform: Render free tier (web service, auto-deploy from GitHub)
