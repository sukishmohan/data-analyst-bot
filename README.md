# AI Data Analyst Agent

An intelligent AI agent system that converts natural language questions into data analysis, charts, and business insights. Powered by NVIDIA NIM API.

## Architecture

```
User Query
    │
    ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ Query Parser │───►│   Planner    │───►│ Code Generator│
└─────────────┘    └──────────────┘    └───────┬───────┘
                                                │
                                                ▼
┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│   Insight    │◄───│ Reflection   │◄───│   Executor    │
│  Generator   │    │    Loop      │    │  (+ Retry)    │
└──────┬───────┘    └──────────────┘    └───────────────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐
│ Visualization│    │   Report     │
│    Module    │    │  Exporter    │
└──────────────┘    └──────────────┘
```

## Quick Start

### Backend

```bash
pip install -r requirements.txt
set NVIDIA_API_KEY=nvapi-...
python server.py
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | API health + LLM status |
| GET | `/api/profile` | Dataset profile |
| POST | `/api/analyze` | Run analysis query |
| GET | `/api/charts/:id` | Serve chart image |
| GET | `/api/sample-queries` | List sample queries |

## Project Structure

```
├── server.py                # FastAPI backend
├── main.py                  # Analysis pipeline orchestrator
├── data_loader.py           # CSV loading, profiling, cleaning
├── executor.py              # Safe code execution with retry
├── visualization.py         # Chart generation
├── forecaster.py            # Time-series forecasting
├── report_exporter.py       # PDF/text report export
├── utils.py                 # LLM client, config, helpers
├── agents/
│   ├── query_parser.py      # NL → structured intent
│   ├── planner.py           # Intent → execution plan
│   ├── code_generator.py    # Plan → Pandas code
│   └── insight_generator.py # Results → insights + reflection
├── frontend/                # React UI (Vite)
└── Sample - Superstore.csv  # Sample dataset
```
