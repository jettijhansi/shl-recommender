# SHL Assessment Recommender — Conversational AI API

A FastAPI service that takes recruiters from vague hiring intent to a precise shortlist of SHL assessments through dialogue.

## Architecture

```
User → POST /chat → Retriever (FAISS) → LLM (Gemini Flash) → JSON Response
```

- **Retriever**: `sentence-transformers` embeds catalog items at startup; FAISS semantic search retrieves top-15 relevant assessments per query.
- **LLM**: Gemini 1.5 Flash with JSON mode for structured, grounded responses.
- **Catalog**: 40 real SHL Individual Test Solutions stored in `catalog.json`.
- **State**: Fully stateless — entire conversation history sent with each request.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key (get one free at https://aistudio.google.com/)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Test it
curl http://localhost:8000/health
# → {"status": "ok"}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to hire a Java developer"}]}'
```

## API Endpoints

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What seniority level are you targeting?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments for a mid-level Java developer with stakeholder interaction needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Rules enforced:**
- `recommendations` is `[]` when still clarifying
- Max 10 items in recommendations
- Max 8 conversation turns
- Only SHL catalog URLs returned
- Off-topic / prompt injection attempts are refused

## Deployment on Render (Free Tier)

1. Push this project to a GitHub repository
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GEMINI_API_KEY` = your key
6. Deploy

Your API will be live at `https://your-app-name.onrender.com`

## Project Structure

```
shl-recommender/
├── main.py           # FastAPI app, endpoints, request/response handling
├── retriever.py      # FAISS index, sentence-transformers, semantic search
├── prompts.py        # System prompt, RAG prompt assembly, conversation formatting
├── catalog.json      # 40 SHL Individual Test Solutions with metadata
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
└── README.md         # This file
```

## Free LLM API Key

Get a free Gemini API key at: https://aistudio.google.com/app/apikey
The free tier (Gemini 1.5 Flash) is sufficient for this project.

## Conversation Behaviors Supported

| Behavior | Example |
|----------|---------|
| Clarify vague queries | "I need an assessment" → asks for role + level |
| Recommend assessments | After 2-3 turns, returns 1-10 SHL assessments |
| Refine mid-conversation | "Add personality tests" → updates shortlist |
| Compare assessments | "Difference between OPQ and Verify G+?" → grounded answer |
| Stay in scope | Refuses HR/legal advice, prompt injection |
| Turn cap | Ends gracefully at 8 turns with final recommendations |

## Evaluation Methodology

The conversational SHL assessment recommender was evaluated across four major dimensions:

### 1. Retrieval Quality
The FAISS semantic retriever was tested using multiple hiring-related queries across different domains and seniority levels.

Evaluation checks included:
- Whether retrieved assessments matched the intended job role
- Relevance of top-k retrieved results
- Consistency of retrieval across similar queries

Example queries tested:
- "Hire a Java backend developer"
- "Need leadership assessment for managers"
- "Assessment for customer support executives"

### 2. Recommendation Relevance
The LLM-generated recommendations were evaluated for:
- Role alignment
- Seniority alignment
- Skill relevance
- Personality/cognitive fit relevance

Recommendations were manually verified against SHL catalog metadata.

### 3. Groundedness and Hallucination Prevention
To ensure grounded responses:
- All recommendations were validated against the internal SHL catalog
- Only catalog-backed URLs were returned
- Hallucinated recommendations were filtered automatically
- Recommendation names and URLs were cross-checked before response generation

### 4. Conversational Effectiveness
The API was tested for:
- Clarification question handling
- Refinement of recommendations after new constraints
- Assessment comparison handling
- Prompt injection rejection
- Off-topic request rejection
- Turn-limit enforcement

### Example Evaluation Scenarios

| Scenario | Expected Behavior |
|---|---|
| Vague hiring request | Ask clarifying questions |
| Add new constraints | Refine recommendations |
| Compare assessments | Return grounded comparison |
| Prompt injection attempt | Reject unsafe request |
| Long conversation | Enforce turn limit |

### Manual Testing
The API was manually tested through:
- FastAPI Swagger UI
- POST /chat endpoint
- Multi-turn conversational scenarios
- Retrieval validation against SHL catalog entries
