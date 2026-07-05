import streamlit as st
import pandas as pd
import datetime
import ast
import json
import re
import requests
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Clinical Trial Prediction Suite",
    page_icon="🧪",
    layout="wide",
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

st.title("🧪 Clinical Trial Feasibility & Visual Forecasting Suite")
st.caption("Hosted inference edition for Hugging Face Spaces free CPU hosting.")

with st.sidebar:
    st.header("Deployment Mode")
    st.write("This build is adapted for free hosting by using remote inference instead of local torch model loading.")
    api_enabled = bool(st.secrets.get("HF_API_KEY", "")) if hasattr(st, "secrets") else False
    st.metric("HF API Key Configured", "Yes" if api_enabled else "No")
    st.caption("If no API key is configured, the app falls back to deterministic demo logic so all features remain usable.")


def build_headers():
    token = ""
    try:
        token = st.secrets.get("HF_API_KEY", "")
    except Exception:
        token = ""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def extract_text_from_response(data):
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first.get("generated_text") or first.get("summary_text") or str(first)
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "error" in data:
            raise RuntimeError(data["error"])
    return str(data)


def call_hf_inference(system_prompt, user_prompt, max_new_tokens=500, temperature=0.2):
    headers = build_headers()
    combined_prompt = f"""System:
{system_prompt}

User:
{user_prompt}

Assistant:
"""
    payload = {
        "inputs": combined_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True
        }
    }
    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    return extract_text_from_response(r.json()).strip()


def severity_score(text):
    t = (text or "").lower()
    score = 0
    patterns = {
        "increase minimum age": 2,
        "exclude": 2,
        "require": 1,
        "hospitalization": 2,
        "cardiovascular": 2,
        "hba1c": 1,
        "narrow": 1,
        "restriction": 1,
        "ecg": 1,
        "washout": 1,
    }
    for k, v in patterns.items():
        if k in t:
            score += v
    nums = re.findall(r"\b\d+\b", t)
    score += min(len(nums), 3)
    return max(score, 1 if t.strip() else 0)


def heuristic_adjustment(df, restrictions):
    sev = severity_score(restrictions)
    updated_records = []
    report_lines = [
        "- Heuristic fallback mode is active because hosted inference is unavailable.",
        f"- Detected protocol restriction severity score: {sev}.",
        "- Sites with weaker baseline enrollment and higher screen failure are penalized more heavily.",
    ]
    for _, row in df.iterrows():
        baseline = float(row["baseline_enrollment_rate"])
        sf = float(row["screen_failure_rate"])
        penalty = min(0.08 * sev + sf * 0.35, 0.65)
        pred_rate = round(max(baseline * (1 - penalty), 0.2), 2)
        pred_sf = min(round(sf + penalty * 0.18, 3), 0.95)
        ratio = pred_rate / baseline if baseline > 0 else 0
        if ratio >= 0.90 and pred_sf < 0.20:
            health_status = "Excellent"
        elif ratio >= 0.75 and pred_sf < 0.30:
            health_status = "Good"
        elif ratio >= 0.55 or pred_sf < 0.45:
            health_status = "Fair"
        else:
            health_status = "Poor"
        updated_records.append({
            "site_id": row["site_id"],
            "site_name": row["site_name"],
            "baseline_enrollment_rate": baseline,
            "screen_failure_rate": sf,
            "predicted_baseline_rate": pred_rate,
            "predicted_screen_failure_rate": pred_sf,
            "site_health": health_status,
        })
    report_lines.append("- Recommendation: widen eligibility criteria selectively at lower-performing sites to reduce projected slippage.")
    return pd.DataFrame(updated_records), "\n".join(report_lines)


def parse_adjusted_rates(response, df_to_predict):
    try:
        data_part, report_part = response.split("---", 1)
        cleaned = data_part.strip().replace("```python", "").replace("```", "")
        adjusted_rates = ast.literal_eval(cleaned)
        return adjusted_rates, report_part.strip()
    except Exception:
        fallback_rates = {
            row["site_id"]: round(row["baseline_enrollment_rate"] * 0.8, 2)
            for _, row in df_to_predict.iterrows()
        }
        return fallback_rates, response.strip()


def run_ai_prediction(df_to_predict, protocol_restrictions):
    if not protocol_restrictions.strip():
        st.warning("Please enter protocol restrictions before generating a prediction.")
        return df_to_predict

    site_summary = df_to_predict[["site_id", "site_name", "baseline_enrollment_rate", "screen_failure_rate"]].to_dict(orient="records")

    system_prompt = (
        "You are a clinical trial feasibility assistant. "
        "Analyze the site metrics and protocol restrictions. "
        "You must provide your response in two strict parts separated by '---':"
        "Part 1: Provide a raw python dictionary mapping site IDs to estimated new enrollment rates based strictly on your logical analysis of the restrictions. Format exactly like this, with NO formatting or markdown backticks:"
        "{'SITE-001': value, 'SITE-002': value, 'SITE-003': value}"
        "---"
        "Part 2: Provide your qualitative professional assessment report with bullet points detailing performance risks."
    )

    user_prompt = f"""Baseline site data:
{site_summary}

New Protocol Restrictions:
{protocol_restrictions}

Evaluate the data and generate both the dynamic rate dictionary and text report."""

    try:
        with st.spinner("AI Engine running inference over tweaked baseline metrics..."):
            response = call_hf_inference(system_prompt, user_prompt, max_new_tokens=500, temperature=0.2)
        adjusted_rates, report_part = parse_adjusted_rates(response, df_to_predict)
    except Exception:
        heuristic_df, report_part = heuristic_adjustment(df_to_predict, protocol_restrictions)
        st.session_state.site_report_text = report_part.strip()
        return heuristic_df

    updated_records = []
    for _, row in df_to_predict.iterrows():
        s_id = row["site_id"]
        pred_rate = float(adjusted_rates.get(s_id, row["baseline_enrollment_rate"] * 0.8))
        drop_ratio = pred_rate / row["baseline_enrollment_rate"] if row["baseline_enrollment_rate"] > 0 else 1
        pred_sf = min(round(row["screen_failure_rate"] * (2.0 - drop_ratio), 3), 0.95)

        if drop_ratio >= 0.90 and pred_sf < 0.20:
            health_status = "Excellent"
        elif drop_ratio >= 0.75 and pred_sf < 0.30:
            health_status = "Good"
        elif drop_ratio >= 0.55 or pred_sf < 0.45:
            health_status = "Fair"
        else:
            health_status = "Poor"

        updated_records.append({
            "site_id": row["site_id"],
            "site_name": row["site_name"],
            "baseline_enrollment_rate": row["baseline_enrollment_rate"],
            "screen_failure_rate": row["screen_failure_rate"],
            "predicted_baseline_rate": round(pred_rate, 2),
            "predicted_screen_failure_rate": pred_sf,
            "site_health": health_status
        })

    st.session_state.site_report_text = report_part.strip()
    return pd.DataFrame(updated_records)


def run_diagnostic_report(user_diagnostic_query, protocol_restrictions, df_breakdown, edited_df_tab3):
    current_metrics_str = df_breakdown[["Site ID", "Site Location", "Timeline Slippage (Days)", "Current Health"]].to_dict(orient="records")
    raw_input_str = edited_df_tab3[["site_id", "baseline_enrollment_rate", "screen_failure_rate", "predicted_baseline_rate"]].to_dict(orient="records")

    diagnostic_sys_prompt = """You are an expert clinical trial auditor. You must execute a structured 5 Whys Root-Cause Analysis.

You are strictly required to format your output using this exact structural layout for the analysis block. Do not use generic markdown text blocks:

**Why 1:** [State the primary visible operational symptom here]
*Answer 1:* [Provide the empirical data factor causing this symptom]

**Why 2:** [Ask why the data factor in Answer 1 occurred]
*Answer 2:* [Explain the local site bottleneck causing it]

**Why 3:** [Ask why the bottleneck in Answer 2 exists]
*Answer 3:* [Connect this directly to the protocol selection issues]

**Why 4:** [Ask why the protocol selection issue occurred]
*Answer 4:* [Explain the clinical friction or patient pool restriction constraint]

**Why 5:** [Ask why the pool constraint is blocking progress]
*Answer 5:* [Provide the core, fundamental root cause of the site failure status]

**Systemic Root Cause Summary:** [Summarize findings in one short sentence]
**Operational Action Item:** [Provide one actionable adjustment solution]"""

    diagnostic_user_prompt = f"""Active User Question: {user_diagnostic_query}

Protocol Restrictions Applied: {protocol_restrictions}

Calculated Operational Context:
{current_metrics_str}

Raw Spreadsheet Variables:
{raw_input_str}"""

    try:
        with st.spinner("Analyzing metrics and conducting 5 Whys Root-Cause Audit..."):
            return call_hf_inference(diagnostic_sys_prompt, diagnostic_user_prompt, max_new_tokens=550, temperature=0.2)
    except Exception:
        sev = severity_score(protocol_restrictions)
        worst = df_breakdown.sort_values("Current Health").iloc[0].to_dict() if len(df_breakdown) else {}
        site_name = worst.get("Site Location", "the selected site")
        return f"""**Why 1:** Why is {site_name} under pressure?
*Answer 1:* Enrollment velocity is lagging while projected slippage remains elevated.

**Why 2:** Why is enrollment lagging?
*Answer 2:* The adjusted eligibility criteria reduce the number of qualifying patients.

**Why 3:** Why does that matter locally?
*Answer 3:* This site starts from a smaller practical recruiting cushion than the stronger sites.

**Why 4:** Why did the protocol create friction?
*Answer 4:* The restrictions combine clinical exclusions with tighter candidate matching requirements.

**Why 5:** Why does the candidate pool shrink so much?
*Answer 5:* The overall restriction severity score is {sev}, which signals materially narrower recruitment feasibility.

**Systemic Root Cause Summary:** Recruitment friction is being driven by stricter eligibility on a limited patient pool.
**Operational Action Item:** Prioritize broader pre-screening and rebalance recruitment toward stronger-performing sites."""


protocol_restrictions = st.text_area(
    "Protocol Recruitment Restrictions / Exclusion Criteria",
    height=100,
    placeholder="""Example:
- Increase minimum age to 55
- Require HbA1c between 7.5 and 9.0
- Exclude patients with recent cardiovascular hospitalization""",
)

if "site_impact_df" not in st.session_state:
    st.session_state.site_impact_df = pd.DataFrame([
        {"site_id": "SITE-001", "site_name": "Ahmedabad Research Center", "baseline_enrollment_rate": 5.2, "screen_failure_rate": 0.18, "predicted_baseline_rate": 4.10, "predicted_screen_failure_rate": 0.22, "site_health": "Good"},
        {"site_id": "SITE-002", "site_name": "Surat Clinical Institute", "baseline_enrollment_rate": 3.9, "screen_failure_rate": 0.24, "predicted_baseline_rate": 2.80, "predicted_screen_failure_rate": 0.31, "site_health": "Fair"},
        {"site_id": "SITE-003", "site_name": "Vadodara Trial Unit", "baseline_enrollment_rate": 6.1, "screen_failure_rate": 0.15, "predicted_baseline_rate": 5.00, "predicted_screen_failure_rate": 0.18, "site_health": "Excellent"},
    ])

if "site_report_text" not in st.session_state:
    st.session_state.site_report_text = "No prediction report generated yet."
if "ts_val" not in st.session_state:
    st.session_state.ts_val = 490
if "ce_val" not in st.session_state:
    st.session_state.ce_val = 112
if "cv_val" not in st.session_state:
    st.session_state.cv_val = 12


tab1, tab2, tab3 = st.tabs([
    "📊 Per-Site Impact Assessment",
    "🔮 Executive Graphical Timeline",
    "📋 Global Site-Level Breakdown"
])

with tab1:
    st.subheader("Clinical Sites Baseline Registry")
    st.dataframe(st.session_state.site_impact_df, use_container_width=True)
    if st.button("Generate Per-Site Prediction", type="primary", key="btn_site_impact"):
        st.session_state.site_impact_df = run_ai_prediction(st.session_state.site_impact_df, protocol_restrictions)
        st.rerun()
    st.markdown("### AI Assessment Report")
    st.markdown(st.session_state.site_report_text)

with tab2:
    st.subheader("🔮 Executive Graphical Timeline")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.ts_val = st.slider("Target Number of Subjects", min_value=10, max_value=2000, value=st.session_state.ts_val, key="early_ts")
    with c2:
        st.session_state.ce_val = st.slider("Currently Enrolled", min_value=0, max_value=st.session_state.ts_val, value=st.session_state.ce_val, key="early_ce")
    with c3:
        st.session_state.cv_val = st.slider("Current Volatility %", min_value=0, max_value=100, value=st.session_state.cv_val, key="early_cv")

    remaining_needed = max(st.session_state.ts_val - st.session_state.ce_val, 0)
    total_baseline_velocity = st.session_state.site_impact_df["baseline_enrollment_rate"].sum()
    total_predicted_velocity = st.session_state.site_impact_df["predicted_baseline_rate"].sum()
    volatility_factor = 1 + (st.session_state.cv_val / 100.0)
    weeks_baseline = remaining_needed / total_baseline_velocity if total_baseline_velocity > 0 else 0
    weeks_predicted = (remaining_needed / total_predicted_velocity if total_predicted_velocity > 0 else 0) * volatility_factor

    start_date = datetime.date(2026, 7, 4)
    baseline_end = start_date + datetime.timedelta(weeks=weeks_baseline)
    predicted_end = start_date + datetime.timedelta(weeks=weeks_predicted)
    delay_days = max((predicted_end - baseline_end).days, 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Planned Completion (Baseline)", baseline_end.strftime("%b %d, %Y"))
    m2.metric("AI Risk-Adjusted Completion", predicted_end.strftime("%b %d, %Y"))
    m3.metric("Projected Timeline Slippage", f"+{delay_days} Days" if delay_days > 0 else "On Schedule", delta=f"{round(delay_days/7, 1)} wks" if delay_days > 0 else None, delta_color="inverse")

    st.markdown("### Cumulative Enrollment Trajectory Projections")
    forecast_data = []
    max_weeks = int(max(weeks_baseline, weeks_predicted, 4)) + 4
    for w in range(max_weeks):
        target_dt = start_date + datetime.timedelta(weeks=w)
        b_count = min(st.session_state.ce_val + (total_baseline_velocity * w), st.session_state.ts_val)
        forecast_data.append({"Date": target_dt, "Patients": b_count, "Scenario": "Planned Target Timeline"})
        adjusted_velocity = total_predicted_velocity / volatility_factor if volatility_factor else total_predicted_velocity
        p_count = min(st.session_state.ce_val + (adjusted_velocity * w), st.session_state.ts_val)
        forecast_data.append({"Date": target_dt, "Patients": p_count, "Scenario": "AI Risk-Adjusted Projection"})

    df_forecast = pd.DataFrame(forecast_data)
    fig_forecast = px.line(df_forecast, x="Date", y="Patients", color="Scenario", color_discrete_map={"Planned Target Timeline": "#1f77b4", "AI Risk-Adjusted Projection": "#d62728"})
    fig_forecast.add_hline(y=st.session_state.ts_val, line_dash="dash", line_color="green", annotation_text="Target Enrollment Cap")
    fig_forecast.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig_forecast, use_container_width=True)

with tab3:
    st.subheader("📋 Live Interactive Site Breakout & Configuration Dashboard")
    st.info("💡 **Reactive Mode Active:** Modifying either the **Baseline Enrollment Rate** or **Screen Failure Rate** recalculates the metric boundaries, updating both the Site Health status indicator and the comparative charts instantly.")

    edited_df_tab3 = st.data_editor(
        st.session_state.site_impact_df,
        use_container_width=True,
        disabled=["site_id", "site_name", "predicted_baseline_rate", "predicted_screen_failure_rate", "site_health"],
        column_config={
            "site_id": st.column_config.TextColumn("Site ID"),
            "site_name": st.column_config.TextColumn("Investigative Site"),
            "baseline_enrollment_rate": st.column_config.NumberColumn("Baseline Rate (pts/wk)", min_value=0.01, format="%.2f"),
            "screen_failure_rate": st.column_config.NumberColumn("Screen Failure Rate", min_value=0.0, max_value=1.0, format="%.2f"),
            "predicted_baseline_rate": st.column_config.NumberColumn("AI Projected Friction Rate (🔒)", format="%.2f"),
            "predicted_screen_failure_rate": st.column_config.NumberColumn("AI Screen Failure Rate (🔒)", format="%.2f"),
            "site_health": st.column_config.TextColumn("🔬 Calculated Site Health (🔒)"),
        },
        key="site_editor_tab3"
    )

    global_target = st.session_state.ts_val
    global_enrolled = st.session_state.ce_val
    current_date = datetime.date(2026, 7, 4)
    weights = [0.4, 0.25, 0.35]
    breakdown_records = []
    gantt_records = []
    chart_y_sites = []
    chart_x_enrolled = []
    chart_x_target = []

    for idx, row in edited_df_tab3.iterrows():
        s_id = row["site_id"]
        orig = row["baseline_enrollment_rate"]
        pred = row["predicted_baseline_rate"]
        sf_rate = row["screen_failure_rate"]

        if orig <= 1.0 or sf_rate >= 0.40 or orig < pred:
            live_health = "Poor"
        elif orig <= 3.0 or sf_rate >= 0.28:
            live_health = "Fair"
        else:
            efficiency_ratio = pred / orig if orig > 0 else 0
            if efficiency_ratio >= 0.80 and sf_rate <= 0.20:
                live_health = "Excellent"
            elif efficiency_ratio >= 0.65 and sf_rate <= 0.25:
                live_health = "Good"
            else:
                live_health = "Fair"

        edited_df_tab3.at[idx, "site_health"] = live_health
        site_target = int(global_target * weights[idx])
        site_enrolled = int(global_enrolled * weights[idx])
        site_remaining = max(site_target - site_enrolled, 0)
        base_weeks = site_remaining / orig if orig > 0 else 1
        performance_factor = pred / orig if orig > 0 else 0.85
        adjusted_weeks = base_weeks / performance_factor if performance_factor > 0 else base_weeks * 1.5
        date_baseline_completion = current_date + datetime.timedelta(weeks=base_weeks)
        date_ai_completion = current_date + datetime.timedelta(weeks=adjusted_weeks)
        days_variance = max((date_ai_completion - date_baseline_completion).days, 0)
        pct_done = round((site_enrolled / site_target) * 100, 1) if site_target > 0 else 0.0

        breakdown_records.append({
            "Site ID": s_id,
            "Site Location": row["site_name"],
            "Progress Done Today": f"{pct_done}% ({site_enrolled}/{site_target})",
            "Target Completion Date (Planned)": date_baseline_completion.strftime("%b %d, %Y"),
            "AI Projected Completion": date_ai_completion.strftime("%b %d, %Y"),
            "Timeline Slippage (Days)": f"+{days_variance} Days" if days_variance > 0 else "On Schedule",
            "Current Health": live_health,
            "Status Flag": "CRITICAL RISK" if days_variance > 30 or live_health == "Poor" else ("WARNING" if days_variance > 10 else "ON TRACK")
        })

        chart_y_sites.append(row["site_name"])
        chart_x_enrolled.append(site_enrolled)
        chart_x_target.append(site_target)
        gantt_records.append(dict(Task=row["site_name"], Start=current_date.strftime("%Y-%m-%d"), Finish=date_baseline_completion.strftime("%Y-%m-%d"), Track="Planned Track Baseline"))
        gantt_records.append(dict(Task=row["site_name"], Start=date_baseline_completion.strftime("%Y-%m-%d"), Finish=date_ai_completion.strftime("%Y-%m-%d"), Track="AI Friction Delay Extension"))

    st.session_state.site_impact_df = edited_df_tab3
    st.markdown("---")

    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(y=chart_y_sites, x=chart_x_enrolled, name="Patients Enrolled", orientation='h', marker_color='#2ca02c'))
    fig_bars.add_trace(go.Bar(y=chart_y_sites, x=chart_x_target, name="Target Allocation Space", orientation='h', marker_color='#1f77b4'))
    fig_bars.update_layout(barmode='stack', title="Site-Level Patient Volume Progression Stack Overview", template="plotly_white", height=260, yaxis=dict(type='category', automargin=True))
    st.plotly_chart(fig_bars, use_container_width=True)

    df_gantt = pd.DataFrame(gantt_records)
    fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Track", color_discrete_map={"Planned Track Baseline": "#1f77b4", "AI Friction Delay Extension": "#d62728"})
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(title="Predictive Calendar Milestone Schedule Map", template="plotly_white", height=220)
    st.plotly_chart(fig_gantt, use_container_width=True)

    st.subheader("📋 Granular Operational Metrics Summary")
    df_breakdown = pd.DataFrame(breakdown_records)
    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🕵️‍♂️ Interactive AI Root-Cause Diagnostic Assistant (5 Whys)")
    st.caption("Ask specific text questions regarding underperforming flags, such as: *'Why is SITE-002 flagged in Poor health?'*")

    user_diagnostic_query = st.text_input("Enter your diagnostic question here:", key="diagnostic_query_input", placeholder="e.g., Explain the poor health score of SITE-002 based on these numbers")

    if st.button("🔍 Run Diagnostic Report", type="primary", key="trigger_diagnostic"):
        if not user_diagnostic_query.strip():
            st.error("Please enter a question first.")
        else:
            diagnostic_response = run_diagnostic_report(user_diagnostic_query, protocol_restrictions, df_breakdown, edited_df_tab3)
            st.info("🎯 **AI 5 Whys Root-Cause Audit Evaluation:**")
            st.markdown(diagnostic_response)
