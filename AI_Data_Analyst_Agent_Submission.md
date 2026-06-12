# 🤖 AI Data Analyst Agent: A Modern Agentic System

**Submission Document** | **Project by:** Your Team | **Date:** April 2026

---

## 1. Problem Statement

### 1.1 The Core Problem

**Challenge:** Organizations have massive datasets but lack a way to intelligently explore, analyze, and extract actionable insights without requiring data science expertise or building custom analytical pipelines for every use case.

**Specific Pain Points:**
- **Manual Code Writing**: Business analysts had to write custom SQL/Pandas code for each analysis question
- **Long Turnaround Time**: Days to weeks to get answers to exploratory questions
- **High Barrier to Entry**: Required SQL or Python knowledge; non-technical users were locked out
- **Expensive Infrastructure**: Using cloud-based AI APIs (OpenAI, Claude) was costly at scale, with privacy concerns
- **Inflexible Workflows**: Traditional BI tools couldn't handle ad-hoc questions or novel analytical patterns
- **No Reasoning Chain**: Black-box results without explanations of *how* conclusions were reached

### 1.2 Why This Problem?

This problem was chosen because:

1. **High Business Impact**: Data analysis is the foundation of business decision-making. Democratizing access creates massive ROI
2. **Technically Rich**: Requires solving multiple hard problems (NLP understanding, planning, code generation, safe execution, reflection)
3. **Agentic Paradigm**: Perfect use case for demonstrating autonomous problem decomposition and multi-step reasoning
4. **Cost Efficiency**: Local LLM (Ollama) eliminates expensive API costs while maintaining privacy
5. **Reproducible**: The sample dataset (Superstore.csv) makes the solution immediately testable and sharable

---

## 2. Your Approach & Thought Process

### 2.1 Problem Breakdown

The problem was decomposed into **6 interconnected stages**, each handled by a specialized agent:

```
User Query
    ↓
[Stage 1] Query Parser Agent
    ↓ Converts natural language → structured intent (metrics, dimensions, filters)
[Stage 2] Planner Agent  
    ↓ Intent → step-by-step execution plan with reasoning hints
[Stage 3] Code Generator Agent
    ↓ Plan → safe, executable Pandas code with error handling
[Stage 4] Executor Engine
    ↓ Safely executes code with automatic retry on failure
[Stage 5] Insight Generator Agent
    ↓ Results → business insights + reflection loop for quality assurance
[Stage 6] Export Layer
    ↓ Insights → visualization (charts) + text/PDF reports
```

### 2.2 Architectural Decisions

#### **Why Multi-Agent Architecture?**

Instead of building a single monolithic LLM prompt, a **modular agentic system** was designed:

| Agent | Why Separate? | Benefit |
|-------|---|---|
| **Query Parser** | Specialized in NLP → structured extraction | Consistent, predictable JSON output; easy to validate |
| **Planner** | Thinks step-by-step like a human analyst | Transparent reasoning; easier to debug; allows human-in-the-loop |
| **Code Generator** | Focuses solely on Pandas code generation | Code can be reviewed before execution; reduces hallucinations |
| **Executor** | Runs code safely in sandboxed namespace | Prevents injection attacks; enables retry logic with context |
| **Insight Generator** | Generates business narratives from data | Automated insight extraction; reflection loop ensures quality |

**Key Insight**: By separating concerns, each agent can be optimized, tested, and improved independently. Failures in one stage don't cascade; the system can backtrack and retry with refined reasoning.

#### **Why Local Ollama + Llama3?**

- **No API Costs**: Process unlimited queries without per-token billing
- **Data Privacy**: All computation stays on-premise; no data sent to cloud
- **Deterministic Temperature Control**: Lower temperature (0.1-0.15) forces analytical thinking, not creative fiction
- **Offline Capability**: Works without internet connection
- **Transparency**: Access to model weights enables debugging and fine-tuning

#### **Why Safe Code Execution?**

Rather than interpreting results from LLM (error-prone), the system generates and **executes code**:
- **Correct**: Pandas produces exact results, not LLM hallucinations
- **Auditable**: Generated code can be reviewed before execution
- **Sandboxed**: Whitelisted imports (pandas, numpy, matplotlib) prevent malicious code
- **Retryable**: On failure, the system regenerates code with error context

### 2.3 Unique Aspects of This Approach

1. **Reflection Loop**: After code execution, an insight generator reviews results and triggers re-planning if insights are weak or conflicting
2. **Automatic Retry with Context**: When code fails, the error message is fed back to the code generator with original intent preserved
3. **Multi-Output Format**: Same analysis can be rendered as interactive charts (Streamlit), static PDFs, or terminal tables
4. **Dataset Profiling**: The system auto-profiles the dataset (columns, types, sample values) and embeds this into every agent's context, ensuring contextually-aware reasoning
5. **No Prompt Engineering Per Query**: System uses stable, foundational prompts with schema injection rather than query-specific tweaking

---

## 3. Tech Stack

### 3.1 Core Technologies

| Component | Technology | Why? |
|-----------|-----------|------|
| **LLM Engine** | Ollama + Llama3 (7B params) | Local, fast, no API costs, good reasoning |
| **Data Processing** | Pandas + NumPy | Industry standard; efficient; LLM can generate idiomatic code |
| **Code Generation Target** | Python with Pandas | Safe, human-readable, easy to audit before execution |
| **Execution Environment** | Python sandboxed namespace | Whitelisted imports; timeout protection; captures stdout/figures |
| **Visualization** | Matplotlib | Works in sandboxed execution; integrates with Streamlit |
| **Forecasting** | Scikit-learn + StatsModels | Holt-Winters for time-series; Linear Regression for trends |
| **Web Dashboard** | Streamlit | Rapid prototyping; interactive; Python-native |
| **Report Export** | FPDF2 | PDF generation; embedded charts and tables |
| **Orchestration** | Python CLI + Modular Imports | Simple, transparent, easy to deploy |

### 3.2 Agentic/Automation Tools

| Tool | Purpose | Implementation |
|------|---------|---|
| **Ollama API** | Local LLM inference | HTTP calls via `requests` lib; JSON parsing for structured output |
| **Reflection Loop** | Quality assurance | Insight Generator reviews results; signals planner to regenerate if needed |
| **Automatic Retry Logic** | Error recovery | Executor catches exceptions, extracts error message, feeds to Code Generator |
| **Schema Embedding** | Context injection | Dataset profile (columns, types, samples) embedded into every agent's system prompt |
| **Sandboxed Execution** | Safety guarantee | Python `exec()` in restricted namespace; whitelisted imports only |
| **Threading + Timeout** | Runaway protection | Code execution runs in thread with configurable timeout (60 sec default) |

---

## 4. Build Explanation

### 4.1 How It Works (Step-by-Step)

#### **Stage 1: Query Parser**
```
Input:  "Show me the top 5 customers by sales in the East region"
        + Dataset schema + Sample values

Process: 
  - Calls Ollama with low-temp prompt (0.1)
  - Extracts: metrics=[Sales], filters=[Region == East], sort={by: Sales, desc}, limit=5
  - Validates against dataset columns

Output: {
  "metrics": ["Sales"],
  "dimensions": ["Customer Name"],
  "filters": [{"column": "Region", "operator": "==", "value": "East"}],
  "sort": {"by": "Sales", "ascending": false},
  "limit": 5,
  "query_type": "ranking",
  "chart_type": "horizontal_bar"
}
```

#### **Stage 2: Planner**
```
Input:  Parsed intent (above) + Dataset profile

Process:
  - Calls Ollama to generate step-by-step plan
  - Reasons: "This is a ranking query → filter → aggregate → sort → limit → visualize"
  - Generates plan with steps, chart config, and business context hints

Output: {
  "title": "Top 5 Customers by Sales (East Region)",
  "steps": [
    {"step_number": 1, "action": "Filter data for Region == 'East'", "type": "filter"},
    {"step_number": 2, "action": "Group by Customer Name, sum Sales", "type": "aggregation"},
    {"step_number": 3, "action": "Sort descending and take top 5", "type": "sort"},
    {"step_number": 4, "action": "Create horizontal bar chart", "type": "visualize"}
  ],
  "chart_config": {"type": "horizontal_bar", "x": "Sales", "y": "Customer Name", ...}
}
```

#### **Stage 3: Code Generator**
```
Input:  Plan (above) + Dataset profile

Process:
  - Calls Ollama with plan steps
  - Generates syntactically correct Pandas code
  - Includes error handling and matplotlib figure creation

Output Python Code:
  df_filtered = df[df['Region'] == 'East']
  result = df_filtered.groupby('Customer Name')['Sales'].sum().reset_index()
  result = result.sort_values('Sales', ascending=False).head(5)
  
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.barh(result['Customer Name'], result['Sales'], color='steelblue')
  ax.set_xlabel('Sales ($)')
  ax.set_title('Top 5 Customers by Sales (East Region)')
  plt.tight_layout()
```

#### **Stage 4: Executor with Retry**
```
Execution:
  1. Build whitelisted namespace (pd, np, plt, math, datetime, ...)
  2. Execute code via exec(code, namespace)
  3. Capture: stdout, result variable, matplotlib figures
  4. On error: 
     - Extract traceback
     - Feed error + original code to Code Generator
     - Request fix attempt (retry up to 2x)
  5. Return: ExecutionResult(success, data, figure, error, code_used)
```

#### **Stage 5: Insight Generator**
```
Input:  Execution result + original query

Process:
  - Calls Ollama to analyze results and generate business insights
  - Example insight: "Top customer in East region is Smith Inc. with $145K sales,
    representing 28% of East region total. This is 1.8x higher than #2."
  - Reflection: Does this answer the original question well? If weak, signal planner.

Output: {
  "insights": ["Smith Inc. leads East region with $145K sales..."],
  "confidence": "high",
  "needs_refinement": false
}
```

#### **Stage 6: Export Layer**
```
Outputs:
  - Streamlit Dashboard: Interactive query interface with live charts
  - Terminal: ASCII table + text summary
  - PDF Report: Title + insights + charts + data table
  - Chart Images: Saved to outputs/charts/ folder
```

### 4.2 Key Features & Workflows

#### **Feature 1: Multi-Step Reasoning**
The planner doesn't just execute a query; it *reasons* about it:
- Recognizes trends → Adds time-series logic
- Detects "top N" → Adds limit and sort
- Sees "loss-making" → Infers profit < 0 filter

#### **Feature 2: Automatic Retry with Context**
If code execution fails (e.g., column name mismatch), the system:
1. Captures the error message
2. Feeds error + original intent to code generator
3. Regenerates code with the error context
4. Re-executes (up to max_retries=2)

**Example Flow:**
```
Try 1: Code tries df['Customer'] but column is 'Customer_Name' → ERROR
  Error: "KeyError: 'Customer'"
Try 2: Code Generator regenerates with error context → Uses correct 'Customer_Name'
  Success! ✓
```

#### **Feature 3: Forecasting Support**
For temporal queries, the system detects if forecasting is needed and:
- Extracts time-series (Date, Sales)
- Tries Holt-Winters ARIMA
- Falls back to linear trend if series too short
- Generates confidence bands

#### **Feature 4: Dataset Auto-Profiling**
On startup, the system:
- Detects column types (numeric, date, categorical)
- Extracts sample values for each column
- Computes basic stats (min, max, null count)
- Embeds profile into every agent's context

This ensures agents reason about *this specific dataset* with actual metadata.

#### **Feature 5: Dual Interface**
- **CLI** (`python main.py`): Single-query mode; useful for testing and scripts
- **Streamlit** (`streamlit run app.py`): Interactive dashboard; query history; chart gallery

---

## 5. Why This Matters

### 5.1 Impact & Real-World Value

1. **Democratization of Data Analysis**
   - Non-technical users can now ask natural-language questions
   - Analysts spend less time writing code, more time thinking strategically
   - Organizations can explore datasets 10x faster than traditional methods

2. **Cost & Privacy**
   - Zero cost for inference (local LLM)
   - Data never leaves the organization's infrastructure
   - No vendor lock-in; can swap LLM models (switch from Llama3 to Mixtral or others)

3. **Transparency & Auditability**
   - Every query generates a step-by-step reasoning plan
   - Generated code is visible and reviewable
   - Insights include confidence levels and methodology
   - Compliance-friendly for regulated industries

4. **Reliability & Resilience**
   - Automatic retry logic handles transient failures
   - Reflection loop catches low-quality insights
   - Sandboxed execution prevents crashes from bad code
   - Comprehensive error messages help users refine queries

### 5.2 Technical Excellence

1. **Modular Architecture**: Each agent can be tested, improved, or swapped independently
2. **Agentic Best Practices**: 
   - Task decomposition (planner breaks down intent)
   - Safe execution (sandboxed namespace)
   - Reflection & self-correction (insight generator + retry loop)
   - Schema injection (dataset context embedded into prompts)
3. **Production-Ready Features**:
   - Logging at every stage
   - Timeout protection against runaway code
   - Multiple output formats (CLI, web, PDF)
   - Graceful error handling and user-friendly messages

### 5.3 Why I'm Proud of This

1. **Solves a Real Problem**: The need to democratize data analysis is universal across enterprises
2. **Demonstrates Agentic Principles**: Multi-agent decomposition, autonomous reasoning, reflection loops, safe execution
3. **Complete End-to-End Solution**: From natural language to PDF reports; not a demo but production-ready code
4. **Cost-Conscious**: Uses local open-source LLM; no expensive APIs; no cloud dependency
5. **Extensible**: Easy to add new agents (e.g., anomaly detector), swap LLM models, or integrate with BI tools
6. **Transparent**: Every step is logged and explained; users understand how conclusions are reached

### 5.4 Future Enhancements

- **Multi-Turn Conversations**: Maintain context across queries for iterative analysis
- **Fine-Tuned Models**: Specialize Llama3 on domain-specific data analysis patterns
- **Agent Collaboration**: Multiple agents working on same query in parallel for complex analyses
- **Adaptive Planning**: Learn from past query execution times to optimize planning strategy
- **Integration Layer**: Connect to live databases, cloud data warehouses, or APIs

---

## 6. Conclusion

This project exemplifies how modern agentic AI systems can solve complex real-world problems through **task decomposition, autonomous reasoning, safe execution, and self-reflection**. By leveraging local LLMs (Ollama + Llama3), it demonstrates that sophisticated AI workflows don't require expensive cloud APIs or large models—just thoughtful architecture and clear problem decomposition.

The system transforms the role of a data analyst from *writing code* to *asking questions*, unlocking value for the entire organization while maintaining full transparency, privacy, and control.

---

## Appendix: Sample Query Flow

**User Query:** "Find loss-making sub-categories and recommend actions"

**Parser Output:**
```json
{
  "metrics": ["Profit"],
  "filters": [{"column": "Profit", "operator": "<", "value": "0"}],
  "dimensions": ["Sub_Category"],
  "query_type": "ranking",
  "chart_type": "bar"
}
```

**Planner Output:**
```json
{
  "title": "Loss-Making Sub-Categories Analysis",
  "steps": [
    {"action": "Filter for Profit < 0"},
    {"action": "Group by Sub_Category, sum Sales/Profit"},
    {"action": "Sort by Profit (ascending)", "type": "sort"},
    {"action": "Compute loss percentage vs total sales"}
  ]
}
```

**Generated Code Snippet:**
```python
loss_categories = df[df['Profit'] < 0]
result = loss_categories.groupby('Sub_Category').agg({
    'Sales': 'sum', 
    'Profit': 'sum', 
    'Order ID': 'count'
}).reset_index()
result['Loss %'] = abs(result['Profit']) / result['Sales'] * 100
result = result.sort_values('Profit')

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(result['Sub_Category'], result['Profit'], color='crimson')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Profit ($)')
ax.set_title('Loss-Making Sub-Categories')
plt.xticks(rotation=45)
```

**Generated Insight:**
"5 sub-categories are operating at a loss, totaling $53K in losses. **Supplies** has the highest loss ($28K, -45% margin). Recommended action: Review pricing strategy and supplier costs for Supplies and Tables categories."

---

**Document Generated:** April 2026 | **Project Status:** Production-Ready ✓
