import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Dynamic EDA Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 800px at 10% -10%, #2e1f4f 0%, #0b1221 55%, #070a12 100%);
        color: #e5e7eb;
    }
    .block-container { padding-top: 0.8rem; padding-bottom: 1.4rem; }
    .card {
        background: rgba(20, 24, 40, 0.78);
        border: 1px solid rgba(120, 130, 180, 0.18);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.28);
        backdrop-filter: blur(6px);
        margin-bottom: 0.6rem;
    }
    .card-title { font-size: 0.9rem; color: #b7c0d1; }
    .card-value { font-size: 1.6rem; font-weight: 700; color: #f8fafc; }
    .subtle { color: #a3acc2; font-size: 0.85rem; }
    .header {
        background: linear-gradient(120deg, rgba(49, 55, 90, 0.9), rgba(18, 22, 38, 0.9));
        border: 1px solid rgba(120, 130, 180, 0.2);
        border-radius: 18px;
        padding: 16px 20px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.3);
    }
    .chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(45, 125, 205, 0.2);
        color: #8fd3ff;
        border: 1px solid rgba(60, 140, 220, 0.4);
        font-size: 0.75rem;
        margin-left: 8px;
    }
    .section-title { font-size: 1.05rem; font-weight: 600; margin: 0.2rem 0 0.5rem 0; }
    .kpi-row { margin-top: 0.2rem; margin-bottom: 0.6rem; }
    .kpi-icon {
        display: inline-flex;
        width: 28px;
        height: 28px;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        margin-right: 8px;
        background: rgba(65, 80, 140, 0.35);
        color: #b8c7ff;
        font-size: 0.8rem;
        border: 1px solid rgba(110, 130, 190, 0.35);
    }
    .kpi-top {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        margin-left: 6px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .badge-low { background: rgba(72, 199, 142, 0.18); color: #9cf4cc; }
    .badge-medium { background: rgba(255, 193, 7, 0.18); color: #ffe28a; }
    .badge-high { background: rgba(255, 107, 107, 0.18); color: #ffb3b3; }
    .tooltip { border-bottom: 1px dotted #8fa0bf; cursor: help; }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start.")
    st.stop()

df = pd.read_csv(uploaded_file)
dataset_name = uploaded_file.name
dataset_valid = (df.shape[0] > 0) and (df.shape[1] > 0)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

header_left, header_right = st.columns([2.2, 1.2], gap="large")
with header_left:
    st.markdown(
        f"""
        <div class="header">
            <div class="section-title">Dynamic EDA Dashboard</div>
            <div class="subtle">Dataset: <strong>{dataset_name}</strong>
                <span class="chip">Uploaded</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    date_range = st.date_input("Date range", [])
    target_col = st.selectbox("Target column", df.columns.tolist())
    sample_pct = st.slider("Dataset sample %", 10, 100, 100, 5)
    st.markdown("</div>", unsafe_allow_html=True)

sample_n = max(1, int(len(df) * (sample_pct / 100)))
df_sample = df.sample(n=sample_n, random_state=42) if sample_n < len(df) else df.copy()

missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
duplicate_rows = df.duplicated().sum()
mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

st.markdown("<div class='section-title'>KPI Summary</div>", unsafe_allow_html=True)
kpi_cols = st.columns(5, gap="large")

dup_pct = (duplicate_rows / max(1, len(df))) * 100
missing_severity = "low" if missing_pct < 5 else "medium" if missing_pct < 20 else "high"
dup_severity = "low" if dup_pct < 1 else "medium" if dup_pct < 5 else "high"
mem_severity = "low" if mem_usage < 50 else "medium" if mem_usage < 200 else "high"

kpis = [
    {
        "label": "Total Rows",
        "value": f"{df.shape[0]:,}",
        "icon": "TR",
        "tip": "Number of rows in the dataset."
    },
    {
        "label": "Total Columns",
        "value": f"{df.shape[1]:,}",
        "icon": "TC",
        "tip": "Number of columns/features in the dataset."
    },
    {
        "label": "Missing Values %",
        "value": f"{missing_pct:.2f}%",
        "icon": "MV",
        "tip": "Total missing values divided by total cells.",
        "severity": missing_severity
    },
    {
        "label": "Duplicate Rows",
        "value": f"{duplicate_rows:,}",
        "icon": "DR",
        "tip": "Rows that are exact duplicates.",
        "severity": dup_severity
    },
    {
        "label": "Memory Usage",
        "value": f"{mem_usage:.2f} MB",
        "icon": "MU",
        "tip": "Estimated memory usage of the dataset.",
        "severity": mem_severity
    },
]

for col, kpi in zip(kpi_cols, kpis):
    badge = ""
    if "severity" in kpi:
        badge = (
            f"<span class='badge badge-{kpi['severity']}'>"
            f"{kpi['severity'].title()}"
            f"</span>"
        )
    col.markdown(
        "<div class='card'>"
        f"<div class='kpi-top'>"
        f"<div class='kpi-icon'>{kpi['icon']}</div>"
        f"<div class='card-title'>{kpi['label']} "
        f"<span class='tooltip' title='{kpi['tip']}'>[?]</span>{badge}</div>"
        f"</div>"
        f"<div class='card-value'>{kpi['value']}</div>"
        "</div>",
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("### Filter Panel")
    apply_filters = st.toggle("Apply filters to all charts", value=True)
    st.caption(f"Dataset size: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    selected_cols = st.multiselect("Column selector", df.columns.tolist(), default=df.columns.tolist())
    dtype_filter = st.multiselect(
        "Column type selector",
        ["Numeric", "Categorical", "Date/Time", "Text"],
        default=["Numeric", "Categorical"]
    )

    st.markdown("**Row-level filters**")
    numeric_filter_col = st.selectbox("Numeric filter column", numeric_cols + ["(none)"])
    numeric_range = None
    if numeric_filter_col != "(none)":
        min_val = float(df[numeric_filter_col].min())
        max_val = float(df[numeric_filter_col].max())
        numeric_range = st.slider(
            "Numeric range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )

    categorical_filter_col = st.selectbox("Categorical filter column", cat_cols + ["(none)"])
    categorical_values = None
    if categorical_filter_col != "(none)":
        unique_vals = df[categorical_filter_col].fillna("NaN").unique().tolist()
        categorical_values = st.multiselect(
            "Category values",
            unique_vals,
            default=unique_vals[:10]
        )

    sampling_control = st.slider("Sampling percentage", 10, 100, 100, 5)
    reset_filters = st.button("Reset filters")

if reset_filters:
    st.experimental_rerun()

df_filtered = df_sample.copy()

if selected_cols:
    df_filtered = df_filtered[selected_cols]

dtype_keep = []
if "Numeric" in dtype_filter:
    dtype_keep += df_filtered.select_dtypes(include="number").columns.tolist()
if "Categorical" in dtype_filter:
    dtype_keep += df_filtered.select_dtypes(include=["object", "category"]).columns.tolist()
if "Date/Time" in dtype_filter:
    dtype_keep += df_filtered.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
if "Text" in dtype_filter:
    dtype_keep += df_filtered.select_dtypes(include=["string"]).columns.tolist()
if dtype_keep:
    df_filtered = df_filtered[dtype_keep]

if apply_filters:
    if numeric_filter_col != "(none)" and numeric_range is not None and numeric_filter_col in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered[numeric_filter_col] >= numeric_range[0]) &
            (df_filtered[numeric_filter_col] <= numeric_range[1])
        ]
    if categorical_filter_col != "(none)" and categorical_values is not None and categorical_filter_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[categorical_filter_col].fillna("NaN").isin(categorical_values)]

if sampling_control < 100:
    sample_n2 = max(1, int(len(df_filtered) * (sampling_control / 100)))
    df_filtered = df_filtered.sample(n=sample_n2, random_state=42)

numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
cat_cols = df_filtered.select_dtypes(exclude="number").columns.tolist()

st.markdown("<div class='section-title'>EDA Visuals</div>", unsafe_allow_html=True)
grid_left, grid_right = st.columns([1.4, 1], gap="large")

with grid_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Correlation Heatmap")
    if numeric_cols:
        method = st.selectbox("Correlation method", ["Pearson", "Spearman"])
        corr = df_filtered[numeric_cols].corr(method=method.lower())
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = corr.mask(mask)

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr_masked.values,
                x=corr_masked.columns,
                y=corr_masked.index,
                colorscale="RdBu",
                zmin=-1, zmax=1,
                text=np.round(corr_masked.values, 2),
                texttemplate="%{text}",
                hovertemplate="Corr(%{y}, %{x}) = %{z:.2f}<br>Higher magnitude means stronger relationship<extra></extra>"
            )
        )

        strong_pairs = (corr.abs() > 0.7) & (~np.eye(len(corr), dtype=bool))
        strong_pairs = strong_pairs & (~mask)
        rows, cols = np.where(strong_pairs.values)
        if len(cols) > 0:
            heatmap.add_trace(
                go.Scatter(
                    x=corr.columns[cols],
                    y=corr.index[rows],
                    mode="markers",
                    marker=dict(size=10, color="rgba(255, 107, 107, 0.6)", line=dict(width=1, color="#ffb3b3")),
                    hoverinfo="skip"
                )
            )

        heatmap.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=350)
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("No numeric columns for correlation heatmap.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Missing Values Heatmap")
    show_only_missing = st.toggle("Show only columns with missing values", value=True)
    if df_filtered.shape[0] > 0:
        missing_counts = df_filtered.isna().sum().sort_values(ascending=False)
        if show_only_missing:
            missing_counts = missing_counts[missing_counts > 0]
        cols_sorted = missing_counts.index.tolist() if len(missing_counts) > 0 else df_filtered.columns.tolist()
        missing_pct_by_col = (df_filtered[cols_sorted].isna().mean() * 100).round(1)
        display_cols = [f"{c} ({missing_pct_by_col[c]}%)" for c in cols_sorted]

        missing_matrix = df_filtered[cols_sorted].isna().astype(int)
        missing_fig = px.imshow(
            missing_matrix,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        missing_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=280,
            xaxis=dict(tickmode="array", tickvals=list(range(len(display_cols))), ticktext=display_cols)
        )
        st.plotly_chart(missing_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with grid_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Histogram / Distribution")
    hist_col = st.selectbox("Histogram column", numeric_cols) if numeric_cols else None
    if hist_col:
        col1, col2 = st.columns(2)
        with col1:
            bins = st.slider("Bins", 10, 60, 30, 5)
            log_scale = st.toggle("Log scale (Y)", value=False)
        with col2:
            clip_method = st.selectbox("Clip outliers", ["None", "IQR", "Percentile"])
            clip_pct = st.slider("Percentile range", 90, 99, 95, 1) if clip_method == "Percentile" else 95

        series = df_filtered[hist_col].dropna()
        if clip_method == "IQR":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            series = series[(series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)]
        elif clip_method == "Percentile":
            low = (100 - clip_pct) / 2
            high = 100 - low
            series = series[(series >= np.percentile(series, low)) & (series <= np.percentile(series, high))]

        skew_val = series.skew() if len(series) > 2 else np.nan
        st.caption(f"Skewness: {skew_val:.3f} (0 is symmetric)")

        hist = px.histogram(
            series,
            x=hist_col,
            nbins=bins,
            color_discrete_sequence=["#5bc0eb"]
        )
        if log_scale:
            hist.update_yaxes(type="log")
        hist.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
        st.plotly_chart(hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Boxplot with Outliers")
    box_col = st.selectbox("Boxplot column", numeric_cols) if numeric_cols else None
    if box_col:
        group_col = st.selectbox("Group by (optional)", ["(none)"] + cat_cols)
        box = px.box(
            df_filtered,
            x=group_col if group_col != "(none)" else None,
            y=box_col,
            points="outliers",
            color_discrete_sequence=["#f77f00"]
        )
        st.caption(
            "Outliers are shown as points beyond 1.5x IQR from the box edges."
        )
        box.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
        st.plotly_chart(box, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Category & Time Insights</div>", unsafe_allow_html=True)
cat_col, time_col, pie_col = st.columns(3, gap="large")

with cat_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Categorical Frequency")
    bar_col = st.selectbox("Category column", cat_cols) if cat_cols else None
    if bar_col:
        counts = df_filtered[bar_col].fillna("NaN").value_counts().head(20).reset_index()
        counts.columns = [bar_col, "count"]
        bar = px.bar(counts, x=bar_col, y="count", color_discrete_sequence=["#7ae582"])
        bar.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
        st.plotly_chart(bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with time_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Time Series Trend")
    time_col_sel = st.selectbox("Datetime column", datetime_cols) if datetime_cols else None
    if time_col_sel and numeric_cols:
        metric_col = st.selectbox("Metric", numeric_cols)
        df_time = df_filtered.copy()
        df_time[time_col_sel] = pd.to_datetime(df_time[time_col_sel], errors="coerce")
        df_time = df_time.dropna(subset=[time_col_sel])
        line = px.line(df_time, x=time_col_sel, y=metric_col, color_discrete_sequence=["#9d4edd"])
        line.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
        st.plotly_chart(line, use_container_width=True)
    else:
        st.info("Select a datetime column and metric.")
    st.markdown("</div>", unsafe_allow_html=True)

with pie_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Category Contribution")
    pie_cat = st.selectbox("Pie chart column", cat_cols) if cat_cols else None
    if pie_cat:
        pie_data = df_filtered[pie_cat].fillna("NaN").value_counts().head(8).reset_index()
        pie_data.columns = [pie_cat, "count"]
        pie = px.pie(pie_data, names=pie_cat, values="count", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Bold)
        pie.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
        st.plotly_chart(pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Insights</div>", unsafe_allow_html=True)
insight_cols = st.columns(3, gap="large")
insights = []

if missing_pct > 0:
    sev = "high" if missing_pct > 20 else "medium" if missing_pct > 5 else "low"
    insights.append({
        "category": "Data Quality",
        "severity": sev,
        "text": f"Overall missing values are {missing_pct:.1f}%."
    })

if dup_pct > 0:
    sev = "high" if dup_pct > 5 else "medium" if dup_pct > 1 else "low"
    insights.append({
        "category": "Data Quality",
        "severity": sev,
        "text": f"Duplicate rows are {dup_pct:.1f}% of the dataset."
    })

if numeric_cols:
    skew_vals = df_filtered[numeric_cols].skew(numeric_only=True).dropna()
    if not skew_vals.empty:
        top_skew = skew_vals.abs().sort_values(ascending=False).index[0]
        sev = "high" if abs(skew_vals[top_skew]) > 2 else "medium" if abs(skew_vals[top_skew]) > 1 else "low"
        insights.append({
            "category": "Distribution",
            "severity": sev,
            "text": f"Column '{top_skew}' is skewed (skew={skew_vals[top_skew]:.2f})."
        })

if numeric_cols:
    corr_abs = df_filtered[numeric_cols].corr().abs()
    np.fill_diagonal(corr_abs.values, 0)
    max_pair = corr_abs.stack().sort_values(ascending=False)
    if not max_pair.empty and max_pair.iloc[0] > 0.7:
        pair = max_pair.index[0]
        sev = "high" if max_pair.iloc[0] > 0.85 else "medium"
        insights.append({
            "category": "Relationships",
            "severity": sev,
            "text": f"Strong correlation between '{pair[0]}' and '{pair[1]}' (|r|={max_pair.iloc[0]:.2f})."
        })

for idx, col in enumerate(insight_cols):
    if idx < len(insights):
        ins = insights[idx]
        col.markdown(
            f"<div class='card'><div class='subtle'>{ins['category']}"
            f"<span class='badge badge-{ins['severity']}'>{ins['severity'].title()}</span>"
            f"</div><div>{ins['text']}</div></div>",
            unsafe_allow_html=True
        )
    else:
        col.markdown("<div class='card'>No additional insights.</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Comparison Section</div>", unsafe_allow_html=True)
cmp_left, cmp_right = st.columns(2, gap="large")
before_df = df.head(10)
after_df = df_filtered.head(10)
changed_cols = sorted(list(set(df.columns) ^ set(df_filtered.columns)))
stat_cols = list(set(df.select_dtypes(include="number").columns).intersection(df_filtered.select_dtypes(include="number").columns))
summary_stats = None
if stat_cols:
    before_stats = df[stat_cols].describe().loc[["mean", "std", "min", "max"]].T
    after_stats = df_filtered[stat_cols].describe().loc[["mean", "std", "min", "max"]].T
    diff_stats = (after_stats - before_stats).round(4)
    summary_stats = pd.concat(
        {"Before": before_stats.round(4), "After": after_stats.round(4), "Diff": diff_stats},
        axis=1
    )

with cmp_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Before (Raw Data Preview)")
    st.dataframe(before_df, use_container_width=True, height=240)
    st.markdown("</div>", unsafe_allow_html=True)
with cmp_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("After (Filtered/Preview)")
    st.caption("Preview preprocessing effect: filters applied")
    st.dataframe(after_df, use_container_width=True, height=240)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
if changed_cols:
    st.write(f"Changed columns: {', '.join(changed_cols)}")
else:
    st.write("Changed columns: none (structure unchanged)")
if summary_stats is not None:
    st.write("Summary statistics changes (numeric columns)")
    st.dataframe(summary_stats, use_container_width=True, height=240)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Actions</div>", unsafe_allow_html=True)
action_cols = st.columns(3, gap="large")
with action_cols[0]:
    confirm_download = st.checkbox("Confirm download", value=False)
    st.button("Download EDA Report", disabled=not (dataset_valid and confirm_download))
with action_cols[1]:
    st.button("Proceed to Preprocessing", type="primary", disabled=not dataset_valid)
with action_cols[2]:
    confirm_save = st.checkbox("Confirm save", value=False)
    st.button("Save EDA Configuration", disabled=not (dataset_valid and confirm_save))
