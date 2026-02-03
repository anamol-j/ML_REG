import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_preprocessing import DataPreprocessing


st.set_page_config(page_title="Dynamic EDA Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 800px at 10% -10%, #2e1f4f 0%, #0b1221 55%, #070a12 100%);
        color: #e5e7eb;
    }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .card {
        background: rgba(20, 24, 40, 0.85);
        border: 1px solid rgba(120, 130, 180, 0.15);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
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
    .section-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem; }
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
kpis = [
    ("Total Rows", f"{df.shape[0]:,}"),
    ("Total Columns", f"{df.shape[1]:,}"),
    ("Missing Values %", f"{missing_pct:.2f}%"),
    ("Duplicate Rows", f"{duplicate_rows:,}"),
    ("Memory Usage", f"{mem_usage:.2f} MB"),
]
for col, (label, value) in zip(kpi_cols, kpis):
    col.markdown(
        f"<div class='card'><div class='card-title'>{label}</div>"
        f"<div class='card-value'>{value}</div></div>",
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("### Filter Panel")
    selected_cols = st.multiselect("Column selector", df.columns.tolist(), default=df.columns.tolist())
    dtype_filter = st.multiselect(
        "Data type filter",
        ["Numeric", "Categorical", "Date/Time", "Text"],
        default=["Numeric", "Categorical"]
    )
    sampling_control = st.slider("Sampling control", 10, 100, 100, 5)
    reset_filters = st.button("Reset filters")

if reset_filters:
    st.experimental_rerun()

df_filtered = df_sample[selected_cols] if selected_cols else df_sample.copy()

numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
cat_cols = df_filtered.select_dtypes(exclude="number").columns.tolist()

st.markdown("<div class='section-title'>EDA Visuals</div>", unsafe_allow_html=True)
grid_left, grid_right = st.columns([1.4, 1], gap="large")

with grid_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Correlation Heatmap")
    if numeric_cols:
        corr = df_filtered[numeric_cols].corr()
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        heatmap.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=350)
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("No numeric columns for correlation heatmap.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Missing Values Heatmap")
    if df_filtered.shape[0] > 0:
        missing_matrix = df_filtered.isna().astype(int)
        missing_fig = px.imshow(
            missing_matrix,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        missing_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
        st.plotly_chart(missing_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with grid_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Histogram / Distribution")
    hist_col = st.selectbox("Histogram column", numeric_cols) if numeric_cols else None
    if hist_col:
        hist = px.histogram(
            df_filtered,
            x=hist_col,
            nbins=30,
            color_discrete_sequence=["#5bc0eb"]
        )
        hist.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=280)
        st.plotly_chart(hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Boxplot with Outliers")
    box_col = st.selectbox("Boxplot column", numeric_cols) if numeric_cols else None
    if box_col:
        box = px.box(
            df_filtered,
            y=box_col,
            points="outliers",
            color_discrete_sequence=["#f77f00"]
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
if numeric_cols:
    skewed = df_filtered[numeric_cols].skew(numeric_only=True).sort_values(ascending=False)
    if not skewed.empty:
        insights.append(f"Column '{skewed.index[0]}' is highly skewed.")
if df.isna().sum().max() > 0:
    worst_missing = df.isna().sum().sort_values(ascending=False).index[0]
    missing_pct_col = (df[worst_missing].isna().mean() * 100)
    insights.append(f"Column '{worst_missing}' has {missing_pct_col:.1f}% missing values.")
if numeric_cols:
    corr = df_filtered[numeric_cols].corr().abs()
    corr.values[[range(corr.shape[0])]*2] = 0
    max_pair = corr.stack().sort_values(ascending=False)
    if not max_pair.empty:
        pair = max_pair.index[0]
        insights.append(f"Strong correlation between '{pair[0]}' and '{pair[1]}'.")

for idx, col in enumerate(insight_cols):
    text = insights[idx] if idx < len(insights) else "Add more rules for insights."
    col.markdown(f"<div class='card'>{text}</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Comparison Section</div>", unsafe_allow_html=True)
cmp_left, cmp_right = st.columns(2, gap="large")
with cmp_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Before (Raw Data Preview)")
    st.dataframe(df.head(10), use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)
with cmp_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("After (Preprocessed Placeholder)")
    st.dataframe(df_sample.head(10), use_container_width=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Actions</div>", unsafe_allow_html=True)
action_cols = st.columns(3, gap="large")
action_cols[0].button("Download EDA Report")
action_cols[1].button("Proceed to Preprocessing")
action_cols[2].button("Save EDA Configuration")
