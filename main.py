import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_cols = [
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
    df["long_run"] = df["total_audi"] / df["first_week_audi"]
    df["long_run"] = df["long_run"].clip(upper=20)

    # 네 속성의 원래 단위 값
    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]
    df["롱런 지수"] = df["long_run"]

    # 네 속성 중 결측값이 있거나 첫 주 관객이 0인 영화 제외
    feature_cols = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]
    valid = df[feature_cols].notna().all(axis=1) & (df["first_week_audi"] != 0)
    df = df.loc[valid].copy()

    # 스크린 수와 누적 관객은 상용로그(log10)
    df["스크린 수"] = df["스크린 수"].where(df["스크린 수"] > 0)
    df["누적 관객"] = df["누적 관객"].where(df["누적 관객"] > 0)
    df = df.dropna(subset=["스크린 수", "누적 관객"]).copy()

    df["스크린 수"] = df["스크린 수"].map(lambda x: __import__("math").log10(x))
    df["누적 관객"] = df["누적 관객"].map(lambda x: __import__("math").log10(x))

    return df


df = load_data()

FEATURES = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]

# -----------------------------
# 제목
# -----------------------------
st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행·상영 특성을 바탕으로 k-평균 군집화를 수행합니다.")

# -----------------------------
# 속성 선택
# -----------------------------
selected_features = st.multiselect(
    "묶는 데 사용할 속성을 선택하세요. (2개 이상)",
    FEATURES,
    default=FEATURES
)

if len(selected_features) < 2:
    st.warning("묶는 데 사용할 속성을 2개 이상 선택하세요.")
    st.stop()

# -----------------------------
# K-means
# -----------------------------
X = df[selected_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 군집 번호를 '누적 관객 평균'이 큰 순서로 ㉮, ㉯, ㉰에 대응
cluster_order = (
    df.groupby("cluster_raw")["누적 관객"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

label_map = {
    cluster_order[0]: "㉮",
    cluster_order[1]: "㉯",
    cluster_order[2]: "㉰"
}
df["묶음"] = df["cluster_raw"].map(label_map)

# -----------------------------
# 상단 요약
# -----------------------------
c1, c2 = st.columns(2)
c1.metric("전체 편수", f"{len(load_data()):,}편")
c2.metric("묶은 편수", f"{len(df):,}편")

# 시각화용 데이터에서 스크린/누적 관객을 원래 단위로 복원
viz_df = df.copy()
import math
viz_df["스크린 수_원래"] = 10 ** viz_df["스크린 수"]
viz_df["누적 관객_원래"] = 10 ** viz_df["누적 관객"]

# -----------------------------
# 2차원 산점도
# -----------------------------
st.subheader("2차원 산점도")

axis_options = FEATURES
col_x, col_y = st.columns(2)
with col_x:
    x_axis = st.selectbox("가로축", axis_options, index=0)
with col_y:
    y_axis = st.selectbox(
        "세로축",
        axis_options,
        index=1 if x_axis != axis_options[1] else 0
    )

fig2 = px.scatter(
    viz_df,
    x=x_axis,
    y=y_axis,
    color="묶음",
    hover_name="movieNm",
    hover_data={
        "movieNm": False,
        "묶음": True,
        x_axis: ":.3f",
        y_axis: ":.3f"
    },
    category_orders={"묶음": ["㉮", "㉯", "㉰"]},
    labels={"묶음": "묶음"}
)
fig2.update_traces(marker=dict(size=7))
fig2.update_layout(legend_title_text="묶음", height=600)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3차원 산점도
# -----------------------------
st.subheader("3차원 산점도")

col_x3, col_y3, col_z3 = st.columns(3)
with col_x3:
    x3 = st.selectbox("x축", axis_options, index=0, key="x3")
with col_y3:
    y3 = st.selectbox("y축", axis_options, index=1, key="y3")
with col_z3:
    z3 = st.selectbox("z축", axis_options, index=2, key="z3")

selected_3d = [x3, y3, z3]

if len(set(selected_3d)) < 3:
    st.info("3차원 산점도를 표시하려면 서로 다른 속성 3개를 선택하세요.")
else:
    fig3 = px.scatter_3d(
        viz_df,
        x=x3,
        y=y3,
        z=z3,
        color="묶음",
        hover_name="movieNm",
        hover_data={
            "movieNm": False,
            "묶음": True
        },
        category_orders={"묶음": ["㉮", "㉯", "㉰"]},
        labels={"묶음": "묶음"}
    )
    fig3.update_traces(marker=dict(size=3))
    fig3.update_layout(legend_title_text="묶음", height=700)
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# 묶음별 통계
# -----------------------------
st.subheader("묶음별 통계")

# 현재 df의 로그 변환 전 원래 값으로 평균 계산
stats_df = pd.DataFrame({
    "묶음": df["묶음"],
    "스크린 수": 10 ** df["스크린 수"],
    "누적 관객": 10 ** df["누적 관객"],
    "10위권 일수": df["10위권 일수"],
    "롱런 지수": df["롱런 지수"]
})

stats = (
    stats_df.groupby("묶음")
    .agg(
        편수=("묶음", "size"),
        스크린_수=("스크린 수", "mean"),
        누적_관객=("누적 관객", "mean"),
        십위권_일수=("10위권 일수", "mean"),
        롱런_지수=("롱런 지수", "mean")
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
)

stats.columns = [
    "묶음", "편수", "스크린 수 평균", "누적 관객 평균",
    "10위권 일수 평균", "롱런 지수 평균"
]

st.dataframe(
    stats.style.format({
        "편수": "{:,.0f}",
        "스크린 수 평균": "{:,.1f}",
        "누적 관객 평균": "{:,.0f}",
        "10위권 일수 평균": "{:,.1f}",
        "롱런 지수 평균": "{:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# 묶음별 누적 관객 상위 5편
# -----------------------------
st.subheader("묶음별 누적 관객 상위 5편")

for label in ["㉮", "㉯", "㉰"]:
    top5 = (
        viz_df[viz_df["묶음"] == label]
        .sort_values("누적 관객_원래", ascending=False)
        .head(5)[["movieNm", "누적 관객_원래"]]
        .reset_index(drop=True)
    )
    top5.index = top5.index + 1
    top5.columns = ["영화 제목", "누적 관객"]
    st.markdown(f"**{label}**")
    st.dataframe(
        top5.style.format({"누적 관객": "{:,.0f}"}),
        use_container_width=True
    )
