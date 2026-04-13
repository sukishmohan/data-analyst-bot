# 🤖 AI Data Analyst Agent (Local Ollama Edition)

An intelligent, production-ready AI agent system that simulates how a real data analyst thinks, plans, executes, and explains insights — powered completely locally by Llama3 via Ollama. No paid APIs required!

## 🏗️ Architecture

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

## 📁 Project Structure

```
project/
├── main.py                 # CLI orchestrator (entry point)
├── app.py                  # Streamlit web dashboard
├── data_loader.py          # CSV loading, profiling, auto-cleaning
├── executor.py             # Safe code execution engine with retry
├── visualization.py        # Chart generation and styling
├── forecaster.py           # Time-series forecasting
├── report_exporter.py      # PDF and text report generation
├── utils.py                # Config, Ollama client, helpers
├── agents/
│   ├── query_parser.py     # Natural language → structured intent
│   ├── planner.py          # Intent → step-by-step execution plan
│   ├── code_generator.py   # Plan → executable Pandas code
│   └── insight_generator.py # Results → business insights + reflection
├── requirements.txt
├── Sample - Superstore.csv
└── outputs/
    ├── charts/             # Saved chart images
    └── reports/            # Exported PDF/text reports
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install & Start Ollama

If you don't have Ollama installed, download it from [ollama.com](https://ollama.com).

Start the Ollama server:
```bash
ollama serve
```

Pull the Llama3 model (first time only):
```bash
ollama pull llama3
```

### 3. Place Your Dataset

Ensure `Sample - Superstore.csv` is in the project root directory.

### 4. Run the Agent

**CLI Mode:**
```bash
python main.py
```

**Streamlit Dashboard:**
```bash
streamlit run app.py
```

## 💡 Sample Queries

| Query | What It Does |
|-------|-------------|
| `Show monthly sales trend` | Time-series line chart of monthly sales |
| `Which category has highest profit?` | Category-level profit comparison |
| `Find loss-making sub-categories` | Filters sub-categories with negative profit |
| `Compare regions by revenue` | Regional sales comparison bar chart |
| `Forecast sales for next 6 months` | Holt-Winters / linear forecast with confidence bands |
| `Top 10 customers by sales` | Ranked customer list with horizontal bar chart |
| `Profit margin by category` | Computed profit margin percentages |
| `Quarterly sales growth rate` | Period-over-period growth analysis |

## 🧠 How the Agent Works

1. **Query Parser** — Uses Llama3 to convert your natural language question into a structured JSON intent (metrics, dimensions, filters, chart type)
2. **Planner** — Breaks the intent into a step-by-step execution plan (data prep → aggregation → visualization)
3. **Code Generator** — Writes clean Pandas code following the plan
4. **Executor** — Runs the code in a sandboxed namespace with timeout protection; auto-retries with regenerated code on failure
5. **Reflection Loop** — Evaluates if the result actually answers the question; re-runs if confidence is low
6. **Insight Generator** — Converts raw numbers into business-friendly bullet points
7. **Visualizer** — Saves charts and auto-detects the best chart type
8. **Report Exporter** — Outputs PDF and text reports on demand
