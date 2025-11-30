import sys
import base64
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import streamlit as st
import pandas as pd
import plotly.express as px
import ruptures as rpt
from fpdf import FPDF
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from app.config import (
    AGG_DAILY_FILE,
    RAW_DATA_FILE,
    KPI_COLUMNS,
)


from app.anomaly_engine import build_hybrid_anomalies, load_daily_data
from app.rca_engine import compute_segment_contributions
from app.insights_engine import generate_insight_text, suggest_actions
from app.data_generator import generate_synthetic_kpi_data
from app import generate_reports
from prophet import Prophet

from app.train_models import train_isolation_forest
from app.model_registry import load_model

@st.cache_data
def get_daily_data():
    return load_daily_data(AGG_DAILY_FILE)


@st.cache_data
def get_raw_data():
    return pd.read_csv(RAW_DATA_FILE, parse_dates=["date"])


@st.cache_data
def get_anomaly_data():
    df_daily = get_daily_data()
    return build_hybrid_anomalies(df_daily)


def _compute_kpi_movers(df_daily: pd.DataFrame) -> pd.DataFrame:
    if len(df_daily) < 2:
        return pd.DataFrame()
    prev = df_daily.iloc[-2]
    latest = df_daily.iloc[-1]
    rows = []
    for kpi in KPI_COLUMNS:
        if kpi not in df_daily.columns:
            continue
        prev_val = prev[kpi]
        curr_val = latest[kpi]
        if prev_val:
            pct = (curr_val - prev_val) / prev_val * 100
        else:
            pct = float("nan")
        rows.append({"kpi": kpi, "prev": prev_val, "current": curr_val, "pct_change": pct})
    movers = pd.DataFrame(rows).sort_values("pct_change", ascending=False)
    return movers


def _compute_change_points(series: pd.Series):
    if len(series) < 10:
        return []
    signal = series.values
    algo = rpt.Pelt(model="rbf").fit(signal)
    breaks = algo.predict(pen=5)
    cp_idx = [b - 1 for b in breaks if b - 1 < len(series)]
    return series.index[cp_idx].tolist()


def _train_decision_tree_rca(anom_df: pd.DataFrame):
    df = anom_df.copy()
    if df["is_hybrid_anomaly"].sum() < 3:
        return None, None

    feature_cols = []
    for col in KPI_COLUMNS + ["day_of_week", "day_of_month"]:
        if col in df.columns:
            feature_cols.append(col)

    if not feature_cols:
        return None, None

    df["label"] = df["is_hybrid_anomaly"].astype(int)
    X = df[feature_cols].fillna(0)
    y = df["label"]

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    return clf, feature_cols


def _build_pdf_report(insight_text: str, actions_text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "KPI Anomaly & RCA Report", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.ln(4)
    pdf.multi_cell(0, 6, "Insight Summary:")
    pdf.ln(2)
    pdf.multi_cell(0, 6, insight_text)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 11)
    pdf.multi_cell(0, 6, "Suggested Actions:")
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    pdf.multi_cell(0, 6, actions_text)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes


def _forecast_kpi(df_daily: pd.DataFrame, kpi: str, periods: int = 30) -> pd.DataFrame:

    df_kpi = df_daily[["date", kpi]].rename(columns={"date": "ds", kpi: "y"})
    model = Prophet(daily_seasonality=True, yearly_seasonality=False)
    model.fit(df_kpi)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def _build_cohort_trends(raw_df: pd.DataFrame, kpi: str, dim: str) -> pd.DataFrame:
    if dim not in ["cohort", "channel", "region"]:
        dim = "cohort"
    tmp = raw_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    grp = (
        tmp.groupby(["date", dim])
        .agg({kpi: "sum"})
        .reset_index()
        .rename(columns={dim: "segment"})
    )
    return grp


def _compute_model_scores(df_daily: pd.DataFrame, df_anom: pd.DataFrame, kpi: str):
    results = []
    if "is_injected_anomaly" not in df_daily.columns:
        return pd.DataFrame()

    y_true = df_daily["is_injected_anomaly"].astype(int).values

    z_col = f"{kpi}_z"
    if z_col in df_anom.columns:
        z = df_anom[z_col].values
        z_pred = (abs(z) > 2.5).astype(int)
        results.append({"model": f"{kpi}-ZScore", "f1": f1_score(y_true, z_pred, zero_division=0)})

    if "iforest_is_anomaly" in df_anom.columns:
        if_pred = df_anom["iforest_is_anomaly"].astype(int).values
        results.append({"model": "IsolationForest", "f1": f1_score(y_true, if_pred, zero_division=0)})

    hyb_pred = df_anom["is_hybrid_anomaly"].astype(int).values
    results.append({"model": "Hybrid", "f1": f1_score(y_true, hyb_pred, zero_division=0)})

    return pd.DataFrame(results).sort_values("f1", ascending=False)


def main():
    st.set_page_config(
        page_title="Business KPI Anomaly Detection & RCA",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("KPI Anomaly & RCA")

    missing_files = []
    if not RAW_DATA_FILE.exists():
        missing_files.append(str(RAW_DATA_FILE))
    if not AGG_DAILY_FILE.exists():
        missing_files.append(str(AGG_DAILY_FILE))


    if st.sidebar.button("🔁 Generate fresh synthetic data"):
        generate_synthetic_kpi_data()

        st.sidebar.success("Synthetic data regenerated. Please reload the page.")

        missing_files = []
        if not RAW_DATA_FILE.exists():
            missing_files.append(str(RAW_DATA_FILE))

        if not AGG_DAILY_FILE.exists():
            missing_files.append(str(AGG_DAILY_FILE))


    if missing_files:
        st.error("🚨 Required data files are missing:")

        for f in missing_files:
            st.write("❌", f)

        st.warning("Please click **'Generate fresh synthetic data'** from the sidebar, then reload.")
        st.stop()
    
    # 🔧 Train / Refresh Model Button
    st.sidebar.markdown("### 🔧 Model Maintenance")

    if st.sidebar.button("Train / Refresh Anomaly Model"):
        try:
            train_isolation_forest()
            st.sidebar.success("IsolationForest model trained & saved successfully.")
        except Exception as e:
            st.sidebar.error(f"Model training failed: {e}")



    kpi = st.sidebar.selectbox("Select KPI", KPI_COLUMNS, index=0)

    df_daily = get_daily_data()
    df_raw = get_raw_data()
    df_anom = get_anomaly_data()

    st.title("📊 Business KPI Anomaly Detection & Root-Cause Analysis")

    tabs = st.tabs(
        [
            "📈 KPI Overview",
            "🚨 Anomaly & Severity",
            "🧠 Root-Cause Analysis",
            "📑 Reports Center",       
            "🌳 ML-based RCA (Decision Tree)",
            "🔀 Change Points & Correlation",
            "📅 KPI Forecasting",
            "👥 Cohort Trends",
            "🆚 Date Comparison",
            "🤔 What-if Simulator",
            "🤖 Model Comparison (AutoML-style)",
            "🎞 Anomaly Replay",
            "✅ Actions, Preview & PDF",    
        ]
    )

    # Tab 1 - 📈 KPI Overview
    with tabs[0]:
        st.subheader(f"KPI Trend – {kpi.replace('_', ' ').title()}")
        fig = px.line(df_daily, x="date", y=kpi, title=f"{kpi} over time")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Basic Stats**")
        st.write(df_daily[kpi].describe().to_frame().T)

        st.markdown("### 🔥 Top KPI Movers (Last Day vs Previous Day)")
        movers = _compute_kpi_movers(df_daily)
        
        if movers.empty:
            st.info("Not enough data points yet to compute movers.")
        else:
            st.dataframe(movers.head(5))


    # Tab 2 - 🚨 Anomaly & Severity
    with tabs[1]:
        st.subheader("Hybrid Anomaly Scores & Severity Levels")
        fig2 = px.line(
            df_anom,
            x="date",
            y="hybrid_anomaly_score",
            color="severity",
            title="Hybrid Anomaly Score Over Time (by Severity)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Top Anomalous Days")
        top_n = st.slider("Show top N anomalies", 3, 30, 10)
        cols_to_show = ["date", "hybrid_anomaly_score", "severity", "is_hybrid_anomaly"] + KPI_COLUMNS
        cols_to_show = [c for c in cols_to_show if c in df_anom.columns]
        top_anom = (
            df_anom.sort_values("hybrid_anomaly_score", ascending=False)
            .head(top_n)[cols_to_show]
        )
        st.dataframe(top_anom)

    # Tab 3 - 🧠 Root-Cause Analysis
    with tabs[2]:
        st.subheader("Root-Cause Analysis (Segment-Level)")

        anomaly_dates = df_anom[df_anom["is_hybrid_anomaly"]]["date"].dt.date.unique()
        if len(anomaly_dates) == 0:
            st.info("No anomalies detected by hybrid engine.")
        else:
            selected_date = st.selectbox("Select anomaly date", anomaly_dates)
            target_ts = pd.to_datetime(selected_date)

            segment_cols = ["channel", "region", "cohort"]
            segment_df = compute_segment_contributions(
                df_raw, target_ts, kpi, segment_cols=segment_cols
            )

            st.markdown(f"### Segment contributions for {selected_date}")
            st.dataframe(segment_df.head(20))

            if not segment_df.empty:
                segment_df["segment"] = (
                    segment_df["channel"]
                    + " | "
                    + segment_df["region"]
                    + " | "
                    + segment_df["cohort"]
                )
                fig3 = px.bar(
                    segment_df.head(10),
                    x="segment",
                    y="contribution_pct_of_change",
                    title=f"Top segment contributors to {kpi} change",
                )
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### Auto-generated Insight")
            insight_text = generate_insight_text(df_raw, df_anom, target_ts, kpi)
            st.code(insight_text, language="markdown")

            st.session_state["last_insight_text"] = insight_text
            st.session_state["last_anomaly_date"] = str(selected_date)

    # Tab 4 - 📑 Reports Center
    with tabs[3]:
        st.subheader("📑 Reports Center – Markdown + PDF Bundle")

        st.markdown(
            "Generate executive summary, KPI overview, anomaly report, "
            "RCA summary, and recommendations as **Markdown + PDF**, "
            "then download them individually or as a single ZIP."
        )

        if st.button("🧾 Generate All Reports"):
            try:
                files = generate_reports.run_all_reports()
                exec_pdf = files["executive_summary_pdf"]
                bundle_zip = files["bundle_zip"]

                st.success("Reports generated successfully.")

                # --- Show PDF preview of Executive Summary ---
                pdf_bytes = exec_pdf.read_bytes()
                b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                pdf_iframe = f"""
                <iframe src="data:application/pdf;base64,{b64}"
                        width="100%" height="600"
                        type="application/pdf"></iframe>
                """
                st.markdown("### 👀 Executive Summary PDF Preview")
                st.markdown(pdf_iframe, unsafe_allow_html=True)

                # --- Download buttons ---
                st.download_button(
                    label="⬇ Download Executive Summary PDF",
                    data=pdf_bytes,
                    file_name="executive_summary.pdf",
                    mime="application/pdf",
                )

                zip_bytes = bundle_zip.read_bytes()
                st.download_button(
                    label="⬇ Download Full Reports Bundle (ZIP)",
                    data=zip_bytes,
                    file_name="reports_bundle.zip",
                    mime="application/zip",
                )

            except Exception as e:
                st.error(f"Failed to generate reports: {e}")
                st.info(
                    "Make sure wkhtmltopdf is installed and WKHTMLTOPDF_CMD "
                    "is configured if required."
                )

    # Tab 5 - 🌳 ML-based RCA (Decision Tree)
    with tabs[4]:
        st.subheader("ML-based RCA – Decision Tree Explanation")

        clf, feature_cols = _train_decision_tree_rca(df_anom)
        if clf is None:
            st.info(
                "Not enough anomaly examples yet to train a decision tree. "
                "Generate more data or extend the time period."
            )
        else:
            st.markdown("### Feature Importance in Predicting Anomalies")
            importances = clf.feature_importances_
            fi_df = pd.DataFrame(
                {"feature": feature_cols, "importance": importances}
            ).sort_values("importance", ascending=True)

            fig_imp = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Which features drive anomalies?",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # Tab 6 - 🔀 Change Points & Correlation
    with tabs[5]:
        st.subheader("🔀 Change Point Detection")
        cp_idx = _compute_change_points(df_daily[kpi])
        fig_cp = px.line(df_daily, x="date", y=kpi, title=f"{kpi} with detected change points")
        if cp_idx:
            cp_dates = df_daily.loc[cp_idx, "date"]
            fig_cp.add_scatter(
                x=cp_dates,
                y=df_daily.loc[cp_idx, kpi],
                mode="markers",
                marker=dict(size=10),
                name="Change Point",
            )
        st.plotly_chart(fig_cp, use_container_width=True)

        st.subheader("📊 KPI Correlation Matrix")
        corr = df_daily[KPI_COLUMNS].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="KPI Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Tab 7 - 📅 KPI Forecasting
    with tabs[6]:
        st.subheader("📅 KPI Forecasting")
        periods = st.slider("Forecast horizon (days)", 7, 60, 30, step=1)
        forecast = _forecast_kpi(df_daily, kpi, periods=periods)
        fig_f = px.line(forecast, x="ds", y="yhat", title=f"{kpi} forecast")
        fig_f.add_scatter(x=df_daily["date"], y=df_daily[kpi], mode="lines", name="Actual")
        st.plotly_chart(fig_f, use_container_width=True)

    # Tab 8 - 👥 Cohort Trends
    with tabs[7]:
        st.subheader("👥 Cohort / Segment Trends")
        dim = st.selectbox("Segment by", ["cohort", "channel", "region"], index=0)
        cohort_df = _build_cohort_trends(df_raw, kpi, dim)
        fig_cohort = px.line(
            cohort_df,
            x="date",
            y=kpi,
            color="segment",
            title=f"{kpi} by {dim} over time",
        )
        st.plotly_chart(fig_cohort, use_container_width=True)

    # Tab 9 - 🆚 Date Comparison
    with tabs[8]:
        st.subheader("🆚 Compare Two Dates")

        all_dates = df_daily["date"].dt.date.unique()
        if len(all_dates) < 2:
            st.info("Need at least two days of data for comparison.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                date_a = st.selectbox("Date A", all_dates, index=0)
            with col_b:
                date_b = st.selectbox("Date B", all_dates, index=len(all_dates) - 1)

            da = df_daily[df_daily["date"].dt.date == date_a].iloc[0]
            db = df_daily[df_daily["date"].dt.date == date_b].iloc[0]

            comp_rows = []
            for c in KPI_COLUMNS:
                if c in df_daily.columns:
                    comp_rows.append(
                        {
                            "kpi": c,
                            "date_a": da[c],
                            "date_b": db[c],
                            "delta": db[c] - da[c],
                            "pct_change": ((db[c] - da[c]) / da[c] * 100) if da[c] else float("nan"),
                        }
                    )
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df)

    # Tab 10 - 🤔 What-if Simulator
    with tabs[9]:
        st.subheader("🤔 What-if KPI Simulator")

        latest = df_daily.iloc[-1]
        st.markdown(f"Base day: **{latest['date'].date()}**")

        conv_adj = st.slider("Change conversion rate (%)", -50, 50, 0)
        pay_adj = st.slider("Change payment success rate (%)", -50, 50, 0)
        dau_adj = st.slider("Change DAU (%)", -50, 50, 0)

        base_sessions = latest.get("sessions", None)
        base_dau = latest.get("dau", None)
        base_orders = latest.get("orders", None)
        base_pay_succ = latest.get("payment_success_rate", None)
        base_rev = latest.get("revenue", None)

        if base_rev is None or base_sessions is None or base_orders is None or base_pay_succ is None:
            st.info("Required base KPIs are missing for simulation.")
        else:
            new_dau = base_dau * (1 + dau_adj / 100) if base_dau else base_dau
            base_cr = base_orders / base_sessions if base_sessions else 0
            new_cr = base_cr * (1 + conv_adj / 100)
            new_orders = base_sessions * new_cr
            new_pay_succ = base_pay_succ * (1 + pay_adj / 100)
            new_succ_orders = new_orders * new_pay_succ
            avg_order_val = base_rev / (base_orders * base_pay_succ) if base_orders and base_pay_succ else 0
            new_rev = new_succ_orders * avg_order_val

            st.markdown("### Simulation Results")
            st.write(f"Base revenue: {base_rev:,.0f}")
            st.write(f"Simulated revenue: {new_rev:,.0f}")
            if base_rev:
                st.write(f"Change: {(new_rev - base_rev):,.0f} ({(new_rev - base_rev) / base_rev * 100:.1f}%)")

    # Tab 11 - 🤖 Model Comparison (AutoML-style)
    with tabs[10]:
        st.subheader("🤖 Model Comparison (AutoML-style)")

        if "is_injected_anomaly" not in df_daily.columns:
            st.info("Injected anomaly labels not available in daily data.")
        else:
            scores_df = _compute_model_scores(df_daily, df_anom, kpi)
            if scores_df.empty:
                st.info("Could not compute model scores.")
            else:
                st.markdown("F1 scores vs injected anomaly labels:")
                st.dataframe(scores_df)
                best = scores_df.iloc[0]
                st.success(f"Best-performing detector for {kpi}: **{best['model']}** (F1={best['f1']:.3f})")

    # Tab 12 - 🎞 Anomaly Replay
    with tabs[11]:
        st.subheader("🎞 Anomaly Replay")
        st.markdown("Use the slider to step through each day and inspect anomaly status.")
        idx = st.slider("Day index", 0, len(df_anom) - 1, len(df_anom) - 1)
        row = df_anom.iloc[idx]
        st.write("**Date:**", row["date"].date())
        st.write("**Hybrid score:**", f"{row['hybrid_anomaly_score']:.3f}")
        st.write("**Severity:**", str(row.get("severity", "N/A")))
        st.write("**Is anomaly:**", bool(row["is_hybrid_anomaly"]))
        fig_r = px.line(df_anom, x="date", y="hybrid_anomaly_score")
        fig_r.add_scatter(x=[row["date"]], y=[row["hybrid_anomaly_score"]], mode="markers", name="Selected")
        st.plotly_chart(fig_r, use_container_width=True)

    # Tab 13 - ✅ Actions, Preview & PDF
    with tabs[12]:
        st.subheader("Suggested Actions, Preview & PDF Report")

        selected_date = st.selectbox(
            "Context date",
            df_anom["date"].dt.date.unique(),
            index=len(df_anom) - 1,
        )
        st.markdown(
            f"Focusing on **{kpi.replace('_', ' ').title()}** around **{selected_date}**."
        )

        actions = suggest_actions(kpi)
        insight_text = st.session_state.get(
            "last_insight_text",
            "Run RCA tab to generate insight text."
        )

        if st.button("📄 Generate PDF Report"):
            pdf_bytes = _build_pdf_report(insight_text, actions)

            st.markdown("### 👀 Report Preview")
            preview_md = f'''
**Insight Summary**

```text
{insight_text}
```

**Recommended Actions**

```text
{actions}
```
'''
            st.markdown(preview_md)

            st.download_button(
                label="Download KPI Anomaly & RCA PDF",
                data=pdf_bytes,
                file_name="kpi_anomaly_rca_report.pdf",
                mime="application/pdf",
            )

        st.info(
            "As a Business Analyst, you would share this report with Product, "
            "Engineering, and Marketing teams, and track the KPI response after actions."
        )

if __name__ == "__main__":
    main()
