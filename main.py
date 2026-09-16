import math

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

FEATURES = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]

# 화면에 표시할 묶음 기호
CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]


# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_cols = [
        "first_scrn",
        "first_show",
        "first_date",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 원래 데이터 전체 편수
    total_movies = len(df)

    # 롱런 지수 = 누적 관객 / 첫 주 관객
    # 20을 넘으면 20으로 제한
    df["long_run"] = df["total_audi"] / df["first_week_audi"]
    df["long_run"] = df["long_run"].clip(upper=20)

    # 군집화에 사용할 네 가지 속성
    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]
    df["롱런 지수"] = df["long_run"]

    # 네 속성에 값이 없거나 첫 주 관객이 0인 영화 제외
    valid = (
        df[FEATURES].notna().all(axis=1)
        & (df["first_week_audi"] != 0)
    )

    df = df.loc[valid].copy()

    # 상용로그를 취하기 위해 스크린 수와 누적 관객이 양수인 영화만 사용
    df = df[
        (df["스크린 수"] > 0)
        & (df["누적 관객"] > 0)
    ].copy()

    # 원래 단위 보관
    df["스크린 수_원래"] = df["스크린 수"]
    df["누적 관객_원래"] = df["누적 관객"]

    # 스크린 수와 누적 관객은 상용로그
    df["스크린 수"] = df["스크린 수"].apply(math.log10)
    df["누적 관객"] = df["누적 관객"].apply(math.log10)

    return df, total_movies


df, total_movies = load_data()


# ---------------------------------------------------------
# K-means 함수
# ---------------------------------------------------------
def run_kmeans(data, selected_features, n_clusters):
    scaler = StandardScaler()

    X = data[selected_features].copy()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    return model, scaler, X_scaled, labels


# ---------------------------------------------------------
# 군집 기호 결정
# 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰ ... 부여
# ---------------------------------------------------------
def make_symbol_map(data, labels):
    temp = data.copy()
    temp["cluster_raw"] = labels

    cluster_order = (
        temp.groupby("cluster_raw")["누적 관객_원래"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    symbol_map = {
        cluster: CLUSTER_SYMBOLS[i]
        for i, cluster in enumerate(cluster_order)
    }

    return symbol_map


# ---------------------------------------------------------
# 제목
# ---------------------------------------------------------
st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행·상영 특성을 바탕으로 k-평균 군집화를 수행합니다.")


# ---------------------------------------------------------
# 묶음 수 선택
# ---------------------------------------------------------
n_clusters = st.selectbox(
    "묶음 수",
    options=list(range(2, 8)),
    index=1,
    format_func=lambda x: f"{x}개",
)


# ---------------------------------------------------------
# 묶음에 사용할 속성 선택
# ---------------------------------------------------------
selected_features = st.multiselect(
    "묶는 데 사용할 속성을 선택하세요. (2개 이상)",
    FEATURES,
    default=FEATURES,
)

if len(selected_features) < 2:
    st.warning("묶는 데 사용할 속성을 2개 이상 선택하세요.")
    st.stop()


# ---------------------------------------------------------
# 현재 선택한 묶음 수로 K-means 실행
# ---------------------------------------------------------
model, scaler, X_scaled, current_labels = run_kmeans(
    df,
    selected_features,
    n_clusters,
)

df_result = df.copy()
df_result["cluster_raw"] = current_labels

symbol_map = make_symbol_map(
    df_result,
    current_labels,
)

df_result["묶음"] = df_result["cluster_raw"].map(symbol_map)


# ---------------------------------------------------------
# 전체 편수 / 묶은 편수
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 편수",
        f"{total_movies:,}편",
    )

with col2:
    st.metric(
        "묶은 편수",
        f"{len(df_result):,}편",
    )


# ---------------------------------------------------------
# 시각화용 데이터
# ---------------------------------------------------------
viz_df = df_result.copy()


# ---------------------------------------------------------
# 2차원 산점도
# ---------------------------------------------------------
st.subheader("2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_axis = st.selectbox(
        "가로축",
        FEATURES,
        index=0,
        key="x_axis_2d",
    )

with col_y:
    default_y = 1 if x_axis != FEATURES[1] else 0

    y_axis = st.selectbox(
        "세로축",
        FEATURES,
        index=default_y,
        key="y_axis_2d",
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
        y_axis: ":.3f",
    },
    category_orders={
        "묶음": CLUSTER_SYMBOLS[:n_clusters]
    },
    labels={
        "묶음": "묶음",
    },
)

fig2.update_traces(
    marker=dict(size=7)
)

fig2.update_layout(
    legend_title_text="묶음",
    height=600,
)

st.plotly_chart(
    fig2,
    use_container_width=True,
)


# ---------------------------------------------------------
# 3차원 산점도
# ---------------------------------------------------------
st.subheader("3차원 산점도")

col_x3, col_y3, col_z3 = st.columns(3)

with col_x3:
    x3 = st.selectbox(
        "x축",
        FEATURES,
        index=0,
        key="x_axis_3d",
    )

with col_y3:
    y3 = st.selectbox(
        "y축",
        FEATURES,
        index=1,
        key="y_axis_3d",
    )

with col_z3:
    z3 = st.selectbox(
        "z축",
        FEATURES,
        index=2,
        key="z_axis_3d",
    )

selected_3d = [x3, y3, z3]

if len(set(selected_3d)) < 3:
    st.info(
        "3차원 산점도를 표시하려면 서로 다른 속성 3개를 선택하세요."
    )

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
            "묶음": True,
        },
        category_orders={
            "묶음": CLUSTER_SYMBOLS[:n_clusters]
        },
        labels={
            "묶음": "묶음",
        },
    )

    # 점을 작게 표시
    fig3.update_traces(
        marker=dict(size=3)
    )

    fig3.update_layout(
        legend_title_text="묶음",
        height=700,
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
    )


# ---------------------------------------------------------
# 묶음별 통계
# ---------------------------------------------------------
st.subheader("묶음별 통계")

# 평균은 로그 변환 전 원래 단위로 계산
stats_df = pd.DataFrame({
    "묶음": df_result["묶음"],
    "스크린 수": df_result["스크린 수_원래"],
    "누적 관객": df_result["누적 관객_원래"],
    "10위권 일수": df_result["10위권 일수"],
    "롱런 지수": df_result["롱런 지수"],
})

stats = (
    stats_df
    .groupby("묶음")
    .agg(
        편수=("묶음", "size"),
        스크린_수=("스크린 수", "mean"),
        누적_관객=("누적 관객", "mean"),
        십위권_일수=("10위권 일수", "mean"),
        롱런_지수=("롱런 지수", "mean"),
    )
    .reindex(CLUSTER_SYMBOLS[:n_clusters])
    .reset_index()
)

stats.columns = [
    "묶음",
    "편수",
    "스크린 수 평균",
    "누적 관객 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]

st.dataframe(
    stats.style.format({
        "편수": "{:,.0f}",
        "스크린 수 평균": "{:,.1f}",
        "누적 관객 평균": "{:,.0f}",
        "10위권 일수 평균": "{:,.1f}",
        "롱런 지수 평균": "{:,.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# 묶음별 누적 관객 상위 5편
# ---------------------------------------------------------
st.subheader("묶음별 누적 관객 상위 5편")

for symbol in CLUSTER_SYMBOLS[:n_clusters]:

    top5 = (
        viz_df[viz_df["묶음"] == symbol]
        .sort_values(
            "누적 관객_원래",
            ascending=False,
        )
        .head(5)[
            ["movieNm", "누적 관객_원래"]
        ]
        .reset_index(drop=True)
    )

    top5.index = top5.index + 1

    top5.columns = [
        "영화 제목",
        "누적 관객",
    ]

    st.markdown(f"**{symbol}**")

    st.dataframe(
        top5.style.format({
            "누적 관객": "{:,.0f}",
        }),
        use_container_width=True,
    )


# ---------------------------------------------------------
# 묶음 수에 따른 군집 내 제곱거리 합
# ---------------------------------------------------------
st.subheader("묶음 수에 따른 군집 내 제곱거리 합")

inertia_rows = []

# 묶음 수를 1~7까지 변경
for k in range(1, 8):

    k_model, _, _, _ = run_kmeans(
        df,
        selected_features,
        k,
    )

    inertia_rows.append({
        "묶음 수": k,
        "군집 내 제곱거리 합": k_model.inertia_,
    })

inertia_df = pd.DataFrame(inertia_rows)


# ---------------------------------------------------------
# 꺾은선 그래프
# ---------------------------------------------------------
fig_inertia = px.line(
    inertia_df,
    x="묶음 수",
    y="군집 내 제곱거리 합",
    markers=True,
)

# 현재 선택한 묶음 수 위치에 세로선
fig_inertia.add_vline(
    x=n_clusters,
    line_dash="dash",
)

fig_inertia.update_layout(
    xaxis=dict(
        tickmode="linear",
        dtick=1,
        title="묶음 수",
    ),
    yaxis_title="군집 내 제곱거리 합",
    height=500,
)

st.plotly_chart(
    fig_inertia,
    use_container_width=True,
)


# ---------------------------------------------------------
# 묶음 수별 값과 직전 값 대비 감소량
# ---------------------------------------------------------
inertia_table = inertia_df.copy()

inertia_table["직전 값 대비 감소량"] = (
    inertia_table["군집 내 제곱거리 합"].shift(1)
    - inertia_table["군집 내 제곱거리 합"]
)

inertia_table.columns = [
    "묶음 수",
    "군집 내 제곱거리 합",
    "직전 값 대비 감소량",
]


# 첫 번째 행은 비교할 앞 값이 없으므로 빈칸
def format_decrease(value):
    if pd.isna(value):
        return ""
    return f"{value:,.2f}"


st.dataframe(
    inertia_table.style.format({
        "묶음 수": "{:,.0f}",
        "군집 내 제곱거리 합": "{:,.2f}",
        "직전 값 대비 감소량": format_decrease,
    }),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# 실루엣 점수
# ---------------------------------------------------------
current_silhouette = silhouette_score(
    X_scaled,
    current_labels,
)

st.markdown(
    f"**현재 선택한 묶음 수({n_clusters}개)의 실루엣 점수: "
    f"{current_silhouette:.3f}**  \n"
    "실루엣 점수는 -1에서 1 사이이며, 1에 가까울수록 묶음이 뚜렷하다는 뜻입니다."
)
