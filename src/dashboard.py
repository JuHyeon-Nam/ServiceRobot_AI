"""
dashboard.py — 서비스 로봇 실시간 관제 대시보드 (디지털 트윈)
------------------------------------------------------------
실제 로봇 궤적(replay.parquet)을 가상 플로어에 재생하면서,
학습된 PdM 모델의 진단을 실시간으로 표시하고 고장 예측 시 경고를 띄운다.
예측은 build_replay.py에서 미리 계산됨(모델이 보는 윈도우와 100% 동일).

실행:  cd src && streamlit run dashboard.py
선행:  build_enhanced_dataset.py → build_replay.py
"""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

DATA = "../data/processed"
GREEN, RED = "#22c55e", "#ef4444"
CATEGORY = {"E-ENV": "환경", "E-INF": "인프라", "E-RBT": "로봇본체"}

st.set_page_config(page_title="로봇 관제 센터", layout="wide", page_icon="🤖")


@st.cache_data
def load():
    df = pd.read_parquet(f"{DATA}/replay.parquet")
    mm = json.load(open(f"{DATA}/robot_pdm_enhanced_meta.json", encoding="utf-8"))
    return df, mm["val_acc"]


df, VAL_ACC = load()
robots = sorted(df["robot"].unique())
nmax = int(df["seq"].max())

# ---------- 사이드바 ----------
st.sidebar.title("⚙️ 관제 설정")
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "playing" not in st.session_state:
    st.session_state.playing = True
speed = st.sidebar.slider("재생 속도(프레임/틱)", 1, 10, 3)
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ 재생", use_container_width=True):
    st.session_state.playing = True
if c2.button("⏸ 정지", use_container_width=True):
    st.session_state.playing = False
manual = st.sidebar.slider("프레임(수동)", 0, nmax, st.session_state.frame)
if manual != st.session_state.frame:
    st.session_state.frame = manual
st.sidebar.caption(f"모델 공식 Validation 정확도 **{VAL_ACC*100:.1f}%**")

st_autorefresh(interval=500, key="tick")
if st.session_state.playing:
    st.session_state.frame = (st.session_state.frame + speed) % (nmax + 1)
fr = st.session_state.frame

# ---------- 현재 프레임 상태 ----------
cur = df[df.seq == fr].copy()
cur["iserr"] = cur["pred"] != "정상"
n_warn = int(cur.iserr.sum()); n_ok = len(cur) - n_warn
health = n_ok / max(len(cur), 1) * 100

st.title("🤖 서비스 로봇 실시간 관제 센터")
st.caption("실제 로봇 궤적을 재생하며 학습된 PdM 모델로 30시점 윈도우를 실시간 진단합니다. "
           "마커=로봇(화살표=진행방향, 색=AI진단), hover로 상세.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("가동 로봇", f"{len(cur)} 대")
k2.metric("정상", f"{n_ok} 대")
k3.metric("⚠️ 고장 경고", f"{n_warn} 대")
k4.metric("운영 건전도", f"{health:.0f} %")

left, right = st.columns([2, 1])

# ---------- 플로어 맵 ----------
with left:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-6, y0=-6, x1=106, y1=106,
                  line=dict(color="#334155", width=2), fillcolor="#0b1220")
    for gx in range(0, 101, 20):
        fig.add_shape(type="line", x0=gx, y0=-6, x1=gx, y1=106, line=dict(color="#1e293b", width=1))
        fig.add_shape(type="line", x0=-6, y0=gx, x1=106, y1=gx, line=dict(color="#1e293b", width=1))
    for _, r in cur.iterrows():
        col = RED if r.iserr else GREEN
        fig.add_trace(go.Scatter(
            x=[r.px], y=[r.py], mode="markers+text",
            marker=dict(symbol="arrow", size=24,
                        angle=float(r.degree) if pd.notna(r.degree) else 0,
                        color=col, line=dict(width=1.5, color="white")),
            text=[r.robot], textposition="top center", textfont=dict(color="#cbd5e1", size=10),
            hovertemplate=(f"<b>{r.robot}</b> ({r.deviceType})<br>"
                           f"AI 진단: {r['pred']} ({r['conf']*100:.0f}%)<br>"
                           f"실제 라벨: {r['errorCode']}<extra></extra>"),
            showlegend=False))
    fig.update_layout(height=580, paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
                      xaxis=dict(visible=False, range=[-10, 110]),
                      yaxis=dict(visible=False, range=[-10, 110]),
                      margin=dict(l=0, r=0, t=6, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.progress(fr / max(nmax, 1), text=f"재생 {fr}/{nmax}")

# ---------- 경고 피드 + 상태표 ----------
with right:
    st.subheader("🚨 실시간 고장 경고")
    warns = cur[cur.iserr]
    if warns.empty:
        st.success("현재 전 로봇 정상 가동 중")
    else:
        for _, w in warns.iterrows():
            cat = CATEGORY.get(w["pred"][:5], "이상")
            mark = "🎯 정답" if w["pred"] == w["errorCode"] else (
                "오탐(실제 정상)" if w["errorCode"] == "정상" else f"실제 {w['errorCode']}")
            st.error(f"**{w.robot}** · `{w['pred']}` ({cat}) "
                     f"{w['conf']*100:.0f}%  \n<small>{mark}</small>", icon="⚠️")

    st.subheader("📋 로봇 상태")
    show = cur[["robot", "deviceType", "pred", "conf"]].copy()
    show["conf"] = (show["conf"] * 100).round(0).astype(int).astype(str) + "%"
    show = show.rename(columns={"robot": "로봇", "deviceType": "종류", "pred": "AI진단", "conf": "신뢰도"})
    st.dataframe(show, hide_index=True, use_container_width=True)
