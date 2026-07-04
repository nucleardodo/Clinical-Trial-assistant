import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
import ast
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Clinical Trial Prediction Suite",
    page_icon="🧪",
    layout="wide",
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

st.title("🧪 Clinical Trial Feasibility & Visual Forecasting Suite")
st.caption("Offline predictive engine with professional Plotly interactive visualization layers.")

# ==========================================
# 1. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Loading tokenizers and model weights into memory...")
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Shared UI: Protocol restriction area used by all features
protocol_restrictions = st.text_area(
    "Protocol Recruitment Restrictions / Exclusion Criteria",
    height=100,
    placeholder="Example:\n- Increase minimum age to 55\n- Require HbA1c between 7.5 and 9.0\n- Exclude patients with recent cardiovascular hospitalization",
)

# Initialize master state registry so edits persist across tabs smoothly
if "site_impact_df" not in st.session_state:
    st.session_state.site_impact_df = pd.DataFrame([
        {"site_id": "SITE-001", "site_name": "Ahmedabad Research Center", "baseline_enrollment_rate": 5.2, "screen_failure_rate": 0.18, "predicted_baseline_rate": 4.10, "predicted_screen_failure_rate": 0.22, "site_health": "Good"},
        {"site_id": "SITE-002", "site_name": "Surat Clinical Institute", "baseline_enrollment_rate": 3.9, "screen_failure_rate": 0.24, "predicted_baseline_rate": 2.80, "predicted_screen_failure_rate": 0.31, "site_health": "Fair"},
        {"site_id": "SITE-003", "site_name": "Vadodara Trial Unit", "baseline_enrollment_rate": 6.1, "screen_failure_rate": 0.15, "predicted_baseline_rate": 5.00, "predicted_screen_failure_rate": 0.18, "site_health": "Excellent"},
    ])

# Helper function to run the AI engine using current state data
def run_ai_prediction(df_to_predict):
    if not protocol_restrictions.strip():
        st.warning("Please enter protocol restrictions before generating a prediction.")
        return df_to_predict
        
    site_summary = df_to_predict[["site_id", "site_name", "baseline_enrollment_rate", "screen_failure_rate"]].to_dict(orient="records")

    system_prompt = (
        "You are a clinical trial feasibility assistant. "
        "Analyze the site metrics and protocol restrictions. "
        "You must provide your response in two strict parts separated by '---':\n\n"
        "Part 1: Provide a raw python dictionary mapping site IDs to estimated new enrollment rates based strictly on your logical analysis of the restrictions. Format exactly like this, with NO formatting or markdown backticks:\n"
        "{'SITE-001': value, 'SITE-002': value, 'SITE-003': value}\n\n"
        "---\n"
        "Part 2: Provide your qualitative professional assessment report with bullet points detailing performance risks."
    )

    user_prompt = (
        f"Baseline site data:\n{site_summary}\n\n"
        f"New Protocol Restrictions:\n{protocol_restrictions}\n\n"
        "Evaluate the data and generate both the dynamic rate dictionary and text report."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with st.spinner("AI Engine running inference over tweaked baseline metrics..."):
        model_inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        input_tensor = model_inputs["input_ids"] if isinstance(model_inputs, dict) or hasattr(model_inputs, "input_ids") else model_inputs

        with torch.no_grad():
            outputs = model.generate(
                input_tensor, max_new_tokens=500, do_sample=True, temperature=0.2, top_p=0.9, pad_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_tensor.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    try:
        data_part, report_part = response.split("---", 1)
        adjusted_rates = ast.literal_eval(data_part.strip().replace("```python", "").replace("```", ""))
    except Exception as e:
        adjusted_rates = {row["site_id"]: round(row["baseline_enrollment_rate"] * 0.8, 2) for _, row in df_to_predict.iterrows()}
        report_part = response

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

# Initialize global states if not present
if "ts_val" not in st.session_state: st.session_state.ts_val = 490
if "ce_val" not in st.session_state: st.session_state.ce_val = 112
if "cv_val" not in st.session_state: st.session_state.cv_val = 12

# ==========================================
# OPERATIONAL ANALYSIS TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📊 Per-Site Impact Assessment", 
    "🔮 Executive Graphical Timeline",
    "📋 Global Site-Level Breakdown"
])

# --- TAB 1 ---
with tab1:
    st.subheader("Clinical Sites Baseline Registry")
    st.dataframe(st.session_state.site_impact_df, use_container_width=True)
    if st.button("Generate Per-Site Prediction", type="primary", key="btn_site_impact"):
        st.session_state.site_impact_df = run_ai_prediction(st.session_state.site_impact_df)
        st.rerun()

# --- TAB 2: EXECUTIVE GRAPHICAL TIMELINE ---
with tab2:
    st.subheader("🔮 Executive Graphical Timeline")
    
    # Earliest slider column organization layout
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
    
    # Volatility impact modifier equations
    volatility_factor = 1 + (st.session_state.cv_val / 100.0)
    weeks_baseline = remaining_needed / total_baseline_velocity if total_baseline_velocity > 0 else 0
    weeks_predicted = (remaining_needed / total_predicted_velocity if total_predicted_velocity > 0 else 0) * volatility_factor
    
    start_date = datetime.date(2026, 7, 4)
    baseline_end = start_date + datetime.timedelta(weeks=weeks_baseline)
    predicted_end = start_date + datetime.timedelta(weeks=weeks_predicted)
    delay_days = max((predicted_end - baseline_end).days, 0)
    
    # Original dynamic metric output block
    m1, m2, m3 = st.columns(3)
    m1.metric("Planned Completion (Baseline)", baseline_end.strftime("%b %d, %Y"))
    m2.metric("AI Risk-Adjusted Completion", predicted_end.strftime("%b %d, %Y"))
    m3.metric("Projected Timeline Slippage", f"+{delay_days} Days" if delay_days > 0 else "On Schedule", delta=f"{round(delay_days/7, 1)} wks" if delay_days > 0 else None, delta_color="inverse")
    
    st.markdown("### Cumulative Enrollment Trajectory Projections")
    forecast_data = []
    max_weeks = int(max(weeks_baseline, weeks_predicted, 4)) + 4
    
    for w in range(max_weeks):
        target_dt = start_date + datetime.timedelta(weeks=w)
        
        # Planned Curve
        b_count = min(st.session_state.ce_val + (total_baseline_velocity * w), st.session_state.ts_val)
        forecast_data.append({"Date": target_dt, "Patients": b_count, "Scenario": "Planned Target Timeline"})
        
        # Risk & Volatility Adjusted Curve
        adjusted_velocity = total_predicted_velocity / volatility_factor
        p_count = min(st.session_state.ce_val + (adjusted_velocity * w), st.session_state.ts_val)
        forecast_data.append({"Date": target_dt, "Patients": p_count, "Scenario": "AI Risk-Adjusted Projection"})
        
    df_forecast = pd.DataFrame(forecast_data)
    fig_forecast = px.line(df_forecast, x="Date", y="Patients", color="Scenario", color_discrete_map={"Planned Target Timeline": "#1f77b4", "AI Risk-Adjusted Projection": "#d62728"})
    fig_forecast.add_hline(y=st.session_state.ts_val, line_dash="dash", line_color="green", annotation_text="Target Enrollment Cap")
    fig_forecast.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig_forecast, use_container_width=True)

# --- TAB 3: GLOBAL SITE-LEVEL BREAKDOWN ---
with tab3:
    st.subheader("📋 Live Interactive Site Breakout & Configuration Dashboard")
    st.info("💡 **Reactive Mode Active:** Modifying either the **Baseline Enrollment Rate** or **Screen Failure Rate** recalculates the metric boundaries, updating both the Site Health status indicator and the comparative charts instantly.")

    # Render data spreadsheet editor right here on Tab 3
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
    
    # Process modifications instantly inside the visual rendering block
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

    # Commit updated rows to global session storage state 
    st.session_state.site_impact_df = edited_df_tab3

    st.markdown("---")

    # CHARTS INTERFACE
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(y=chart_y_sites, x=chart_x_enrolled, name="Patients Enrolled", orientation='h', marker_color='#2ca02c'))
    fig_bars.add_trace(go.Bar(y=chart_y_sites, x=chart_x_target, name="Target Allocation Space", orientation='h', marker_color='#1f77b4'))
    
    # FIXED: Added explicit type tracking and automatic margin accommodations to preserve site labels
    fig_bars.update_layout(
        barmode='stack', 
        title="Site-Level Patient Volume Progression Stack Overview", 
        template="plotly_white", 
        height=260,
        yaxis=dict(type='category', automargin=True)
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    df_gantt = pd.DataFrame(gantt_records)
    fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Track", color_discrete_map={"Planned Track Baseline": "#1f77b4", "AI Friction Delay Extension": "#d62728"})
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(title="Predictive Calendar Milestone Schedule Map", template="plotly_white", height=220)
    st.plotly_chart(fig_gantt, use_container_width=True)

    st.subheader("📋 Granular Operational Metrics Summary")
    df_breakdown = pd.DataFrame(breakdown_records)
    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    # INTERACTIVE HEALTH DIAGNOSTIC CHAT
    st.markdown("---")
    st.subheader("🕵️‍♂️ Interactive AI Root-Cause Diagnostic Assistant (5 Whys)")
    st.caption("Ask specific text questions regarding underperforming flags, such as: *'Why is SITE-002 flagged in Poor health?'*")

    user_diagnostic_query = st.text_input("Enter your diagnostic question here:", key="diagnostic_query_input", placeholder="e.g., Explain the poor health score of SITE-002 based on these numbers")

    if st.button("🔍 Run Diagnostic Report", type="primary", key="trigger_diagnostic"):
        if not user_diagnostic_query.strip():
            st.error("Please enter a question first.")
        else:
            current_metrics_str = df_breakdown[["Site ID", "Site Location", "Timeline Slippage (Days)", "Current Health"]].to_dict(orient="records")
            raw_input_str = edited_df_tab3[["site_id", "baseline_enrollment_rate", "screen_failure_rate", "predicted_baseline_rate"]].to_dict(orient="records")
            
            # REINFORCED STRUCTURAL LAYOUT RULES
            diagnostic_sys_prompt = (
                "You are an expert clinical trial auditor. You must execute a structured 5 Whys Root-Cause Analysis.\n"
                "You are strictly required to format your output using this exact structural layout for the analysis block. Do not use generic markdown text blocks:\n\n"
                "**Why 1:** [State the primary visible operational symptom here]\n"
                "*Answer 1:* [Provide the empirical data factor causing this symptom]\n\n"
                "**Why 2:** [Ask why the data factor in Answer 1 occurred]\n"
                "*Answer 2:* [Explain the local site bottleneck causing it]\n\n"
                "**Why 3:** [Ask why the bottleneck in Answer 2 exists]\n"
                "*Answer 3:* [Connect this directly to the protocol selection issues]\n\n"
                "**Why 4:** [Ask why the protocol selection issue occurred]\n"
                "*Answer 4:* [Explain the clinical friction or patient pool restriction constraint]\n\n"
                "**Why 5:** [Ask why the pool constraint is blocking progress]\n"
                "*Answer 5:* [Provide the core, fundamental root cause of the site failure status]\n\n"
                "**Systemic Root Cause Summary:** [Summarize findings in one short sentence]\n"
                "**Operational Action Item:** [Provide one actionable adjustment solution]"
            )
            
            diagnostic_user_prompt = (
                f"Active User Question: {user_diagnostic_query}\n\n"
                f"Protocol Restrictions Applied: {protocol_restrictions}\n\n"
                f"Calculated Operational Context:\n{current_metrics_str}\n\n"
                f"Raw Spreadsheet Variables:\n{raw_input_str}"
            )
            
            chat_messages = [
                {"role": "system", "content": diagnostic_sys_prompt},
                {"role": "user", "content": diagnostic_user_prompt}
            ]
            
            with st.spinner("Analyzing metrics and conducting 5 Whys Root-Cause Audit..."):
                model_inputs = tokenizer.apply_chat_template(
                    chat_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                
                if isinstance(model_inputs, dict):
                    input_tensor = model_inputs["input_ids"].to(model.device)
                elif hasattr(model_inputs, "input_ids"):
                    input_tensor = model_inputs.input_ids.to(model.device)
                else:
                    input_tensor = model_inputs.to(model.device)
                
                with torch.no_grad():
                    gen_outputs = model.generate(
                        input_tensor, 
                        max_new_tokens=550, 
                        temperature=0.2, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                diagnostic_response = tokenizer.decode(
                    gen_outputs[0][input_tensor.shape[-1]:], 
                    skip_special_tokens=True
                ).strip()
            
            st.info("🎯 **AI 5 Whys Root-Cause Audit Evaluation:**")
            st.markdown(diagnostic_response)