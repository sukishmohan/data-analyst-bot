"""
app.py — Streamlit Dashboard for AI Data Analyst Agent (Ollama Edition)

A premium, interactive web UI for the fully-local agent system.
Uses Ollama + Llama3. No paid APIs.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import get_logger, check_ollama_available, check_model_available, OLLAMA_MODEL
from data_loader import load_csv, profile_dataset, auto_clean
from agents.query_parser import parse_query
from agents.planner import generate_plan
from agents.code_generator import generate_code
from agents.insight_generator import generate_insights, reflect
from executor import execute_with_retry, format_result
from visualization import generate_fallback_chart
from report_exporter import export_text_report
from forecaster import forecast_series

log = get_logger("streamlit_app")

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p { color: #a2a8d3; font-size: 1.05rem; margin-top: 0.4rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        border: 1px solid rgba(233, 69, 96, 0.2);
    }
    .metric-card .metric-value { font-size: 1.8rem; font-weight: 700; color: #e94560; }
    .metric-card .metric-label { font-size: 0.85rem; color: #a2a8d3; margin-top: 0.3rem; }

    .pipeline-step {
        background: #f8f9fa;
        border-left: 4px solid #e94560;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    .insight-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }

    .status-ok { color: #28a745; font-weight: 600; }
    .status-err { color: #dc3545; font-weight: 600; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #a2a8d3; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.profile = None
    st.session_state.history = []


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🤖 AI Data Analyst Agent</h1>
    <p>Powered by Llama3 via Ollama — Fully local, no paid APIs</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Ollama Status Check
# ──────────────────────────────────────────────

col_status1, col_status2 = st.columns(2)
with col_status1:
    groq_ok = check_ollama_available()
    if groq_ok:
        st.markdown('<p class="status-ok">✅ Groq API: Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">❌ Groq API: Key not configured in Streamlit Secrets</p>', unsafe_allow_html=True)

with col_status2:
    if groq_ok:
        st.markdown(f'<p class="status-ok">✅ Model {OLLAMA_MODEL}: Ready (Lightning speed ⚡)</p>', unsafe_allow_html=True)



# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📂 Dataset")

    project_dir = Path(__file__).resolve().parent
    csv_files = list(project_dir.glob("*.csv"))

    data_source = st.radio("Data source", ["Auto-detect", "Upload CSV"], index=0)

    if data_source == "Auto-detect":
        if csv_files:
            selected = st.selectbox("Select file", csv_files, format_func=lambda p: p.name)
            if st.button("Load Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading and profiling dataset..."):
                    df = load_csv(selected)
                    profile = profile_dataset(df)
                    df = auto_clean(df, profile)
                    profile = profile_dataset(df)
                    st.session_state.df = df
                    st.session_state.profile = profile
                st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} cols")
        else:
            st.warning("No CSV files found.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            with st.spinner("Processing..."):
                try:
                    df = pd.read_csv(uploaded, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    try:
                        df = pd.read_csv(uploaded, encoding="latin-1")
                    except UnicodeDecodeError:
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded, encoding="cp1252")
                profile = profile_dataset(df)
                df = auto_clean(df, profile)
                profile = profile_dataset(df)
                st.session_state.df = df
                st.session_state.profile = profile
            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} cols")

    # Profile
    if st.session_state.profile is not None:
        p = st.session_state.profile
        st.markdown("---")
        st.markdown("### 📊 Dataset Profile")
        st.markdown(f"**Shape:** {p['shape']['rows']:,} × {p['shape']['columns']}")
        st.markdown(f"**Duplicates:** {p['duplicate_rows']}")
        st.markdown(f"**Missing:** {p['total_missing']}")

        with st.expander("📋 Columns"):
            col_df = pd.DataFrame({
                "Column": p["columns"],
                "Type": [p["dtypes"][c] for c in p["columns"]],
                "Missing": [p.get("missing", {}).get(c, 0) for c in p["columns"]],
            })
            st.dataframe(col_df, use_container_width=True, hide_index=True)

        with st.expander("🔢 Numeric Stats"):
            if "numeric_stats" in p:
                st.dataframe(pd.DataFrame(p["numeric_stats"]), use_container_width=True)

        with st.expander("🏷️ Sample Values"):
            for col, vals in p.get("sample_values", {}).items():
                st.markdown(f"**{col}:** {', '.join(str(v) for v in vals[:6])}")

    st.markdown("---")
    st.markdown("### 💡 Sample Queries")
    sample_queries = [
        "Show monthly sales trend",
        "Which category has highest profit?",
        "Find loss-making sub-categories",
        "Compare regions by revenue",
        "Forecast sales for next 6 months",
        "Top 10 customers by sales",
        "Profit margin by category",
        "Quarterly sales growth rate",
    ]
    for sq in sample_queries:
        if st.button(f"📌 {sq}", key=f"sample_{sq}", use_container_width=True):
            st.session_state["pending_query"] = sq


# ──────────────────────────────────────────────
# Main Area
# ──────────────────────────────────────────────

if st.session_state.df is None:
    st.info("👈 Load a dataset from the sidebar to get started.")
    if csv_files:
        with st.expander("👀 Quick preview"):
            # Use data_loader's robust load_csv to automatically handle latin-1 encoding
            try:
                st.dataframe(load_csv(csv_files[0]).head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load preview: {e}")
    st.stop()

df = st.session_state.df
profile = st.session_state.profile

# Metrics row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{profile["shape"]["rows"]:,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{profile["shape"]["columns"]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(profile.get("numeric_columns", []))}</div><div class="metric-label">Numeric Fields</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(profile.get("categorical_columns", []))}</div><div class="metric-label">Categorical Fields</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📋 Data Preview (first 100 rows)"):
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Query Input
# ──────────────────────────────────────────────

st.markdown("### 🔍 Ask Your Question")

default_query = st.session_state.pop("pending_query", "")

query = st.text_input(
    "Enter your analysis question:",
    value=default_query,
    placeholder="e.g., Show monthly sales trend",
    label_visibility="collapsed",
)

col_run, col_export = st.columns([3, 1])
with col_run:
    run_clicked = st.button("🚀 Run Analysis", type="primary",
                             use_container_width=True, disabled=not query)
with col_export:
    export_report = st.checkbox("📤 Export Report", value=False)


# ──────────────────────────────────────────────
# Analysis Pipeline
# ──────────────────────────────────────────────

if run_clicked and query:
    if not check_ollama_available():
        st.error("❌ Groq API key is missing. Please configure it in Streamlit Secrets.")
        st.stop()

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Pipeline")

    # Step 1: Parse
    with st.status("🧠 Understanding your question...", expanded=True) as status:
        try:
            intent = parse_query(query, profile)
            st.markdown(f"**Intent:** {intent.get('description', query)}")
            st.markdown(f"**Type:** `{intent.get('query_type')}` | **Chart:** `{intent.get('chart_type')}`")
            st.json(intent, expanded=False)
            status.update(label="🧠 Query understood ✅", state="complete")
        except Exception as e:
            status.update(label=f"🧠 Query parsing failed: {e}", state="error")
            st.stop()

    # Step 2: Plan
    with st.status("📋 Creating execution plan...", expanded=True) as status:
        try:
            plan = generate_plan(intent, profile)
            st.markdown(f"**Title:** {plan['title']}")
            for s in plan.get("steps", []):
                st.markdown(f"""<div class="pipeline-step"><strong>Step {s.get('step_number', '?')}</strong> [{s.get('type', '')}] {s.get('action', '')}</div>""", unsafe_allow_html=True)
            status.update(label="📋 Plan created ✅", state="complete")
        except Exception as e:
            status.update(label=f"📋 Planning failed: {e}", state="error")
            st.stop()

    # Step 3: Code generation & execution
    is_forecast = plan.get("needs_forecast") or intent.get("query_type") == "forecast"

    if is_forecast:
        with st.status("🔮 Running forecast...", expanded=True) as status:
            try:
                date_cols = profile.get("date_columns", [])
                date_col = date_cols[0] if date_cols else "Order Date"
                metrics = intent.get("metrics", ["Sales"])
                value_col = metrics[0] if metrics else "Sales"
                forecast_df, fig = forecast_series(df, date_col, value_col, periods=6, freq="MS")
                result_text = forecast_df.to_string()
                exec_fig = fig
                final_code = "(forecast module)"
                status.update(label="🔮 Forecast complete ✅", state="complete")
            except Exception as e:
                status.update(label=f"🔮 Forecast failed: {e}", state="error")
                st.stop()
    else:
        with st.status("💻 Generating code...", expanded=True) as status:
            try:
                code = generate_code(plan, profile)
                st.code(code, language="python")
                status.update(label="💻 Code generated ✅", state="complete")
            except Exception as e:
                status.update(label=f"💻 Code generation failed: {e}", state="error")
                st.stop()

        with st.status("⚙️ Executing...", expanded=True) as status:
            try:
                exec_result, final_code = execute_with_retry(code, df, plan, profile, max_retries=2)
                if not exec_result.success:
                    status.update(label=f"⚙️ Execution failed: {exec_result.error[:200]}", state="error")
                    st.code(exec_result.error, language="text")
                    st.stop()
                result_text = format_result(exec_result)
                exec_fig = exec_result.figure
                st.text(result_text[:2000])
                status.update(label="⚙️ Execution successful ✅", state="complete")
            except Exception as e:
                status.update(label=f"⚙️ Error: {e}", state="error")
                st.stop()

    # Step 4: Reflection
    with st.status("🔄 Validating...", expanded=False) as status:
        try:
            reflection = reflect(query, plan, result_text)
            confidence = reflection.get("confidence", 0)
            st.markdown(f"**Confidence:** {confidence:.0%}")
            status.update(label=f"🔄 Validated ({confidence:.0%}) ✅", state="complete")
        except Exception:
            reflection = {"is_complete": True, "confidence": 0.8}
            status.update(label="🔄 Validation skipped", state="complete")

    # Results
    st.markdown("---")
    st.markdown("### 📊 Results")

    r_col, c_col = st.columns([1, 1])

    with r_col:
        st.markdown("#### 📋 Data Result")
        if is_forecast:
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        elif exec_result.result is not None:
            r = exec_result.result
            if isinstance(r, (pd.DataFrame, pd.Series)):
                if isinstance(r, pd.Series):
                    r = r.reset_index()
                st.dataframe(r.head(50), use_container_width=True, hide_index=True)
            else:
                st.metric("Result", str(r))
        else:
            st.text(result_text[:2000])

    with c_col:
        st.markdown("#### 📈 Visualization")
        fig_display = exec_fig if exec_fig is not None else None
        if fig_display is None and not is_forecast:
            fig_display = generate_fallback_chart(
                exec_result.result, plan.get("chart_config", {}), plan.get("title", "Analysis"))
        if fig_display is not None:
            st.pyplot(fig_display)
        else:
            st.info("No chart generated.")

    # Insights
    st.markdown("---")
    st.markdown("### 💡 Business Insights")

    with st.status("💡 Generating insights...", expanded=True) as status:
        try:
            insights = generate_insights(query, result_text, plan.get("title", ""))
            status.update(label="💡 Insights generated ✅", state="complete")
        except Exception as e:
            insights = "Could not generate insights."
            status.update(label=f"💡 Failed: {e}", state="error")

    st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

    with st.expander("🔧 Generated Code"):
        st.code(final_code, language="python")

    if export_report:
        st.markdown("---")
        st.markdown("### 📤 Reports")
        txt_path = export_text_report(query, plan["title"], plan.get("steps", []), result_text, insights)
        with open(txt_path, "r", encoding="utf-8") as f:
            st.download_button("📥 Download Report", data=f.read(), file_name=txt_path.name, mime="text/plain", use_container_width=True)

    st.session_state.history.append({
        "query": query,
        "title": plan.get("title", query),
        "insights_snippet": insights[:200],
    })

# History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Query History")
    for h in reversed(st.session_state.history):
        with st.expander(f"🕐 {h['title']}", expanded=False):
            st.markdown(f"**Query:** {h['query']}")
            st.markdown(f"**Insights:** {h['insights_snippet']}...")
