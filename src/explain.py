"""
예지보전 진단의 '근거' 설명 모듈 (Explainability)
--------------------------------------------------
LightGBM 내장 SHAP(`pred_contrib=True`)으로 각 피처의 기여도를 구하고,
249개 저수준 피처를 사람이 읽을 수 있는 '물리적 신호 그룹'으로 묶어
"왜 이 진단을 내렸는가"의 Top-K 근거를 반환한다.

- 무거운 `shap` 패키지 불필요 → 모델 경량성(2.8MB·CPU) 유지.
- make_features()의 피처 배치 순서와 1:1로 정합되어야 함(아래 주석 참고).
"""
from __future__ import annotations
import numpy as np

# make_features()에서 동적센서는 MODEL_DYN_IDX=[0,1,4,5,6]만 사용
# dyn 원본=[batteryLevel,speed,x,y,degree,collision,obstacle] → 사용 5종:
DYN5 = ["batteryLevel", "speed", "degree", "collision", "obstacle"]
STAT9 = ["isOffline", "nowCharging", "emergencyStop", "batteryUse",
         "batteryCycleCount", "distance", "crowd", "deviceType", "mainState"]
N_FFT = 15  # make_features: rfft[:, :15, :]

# 사람이 읽는 물리적 그룹명(자소서/관제 화면용)
GROUP_LABEL = {
    "batteryLevel": "배터리 잔량 추이",
    "speed": "이동 속도·진동",
    "degree": "진행각(방향) 동역학",
    "collision": "충돌 신호",
    "obstacle": "장애물 감지",
    "isOffline": "통신 오프라인 여부",
    "nowCharging": "충전 상태",
    "emergencyStop": "긴급정지 신호",
    "batteryUse": "누적 배터리 소모",
    "batteryCycleCount": "배터리 충·방전 사이클",
    "distance": "누적 주행거리(구동부 마모)",
    "crowd": "주변 혼잡도",
    "deviceType": "로봇 종류",
    "mainState": "동작 상태",
}


def build_group_index(n_features: int = 249) -> np.ndarray:
    """피처 인덱스(249) → 그룹 키 배열. make_features 배치 순서와 동일하게 구성."""
    groups: list[str] = []
    # 1) flatten: 30시점 × 5센서  (행우선: t별로 5센서)  → 150
    for _t in range(30):
        for s in DYN5:
            groups.append(s)
    # 2) mean(5) · 3) std(5) · 4) trend(5)
    for _ in range(3):
        for s in DYN5:
            groups.append(s)
    # 5) FFT: 15주파수 × 5센서 (행우선: 주파수별로 5센서) → 75
    for _k in range(N_FFT):
        for s in DYN5:
            groups.append(s)
    # 6) 정적 9
    for s in STAT9:
        groups.append(s)
    arr = np.array(groups)
    assert len(arr) == n_features, f"group index {len(arr)} != n_features {n_features}"
    return arr


_GROUP_IDX: np.ndarray | None = None


def explain_prediction(booster, features: np.ndarray, pred_idx: int,
                       n_classes: int, n_features: int = 249, top_k: int = 3) -> list[dict]:
    """예측 클래스에 대한 피처 기여도를 물리 그룹으로 합산해 Top-K 근거를 반환.

    반환 예: [{"factor": "배터리 잔량 추이", "impact": "+5.21", "effect": "진단을 강하게 뒷받침"}, ...]
    impact 부호 = 해당 신호가 이 진단을 밀어준(+) / 반대로 끌어내린(-) 정도(로짓 기여).
    """
    global _GROUP_IDX
    if _GROUP_IDX is None:
        _GROUP_IDX = build_group_index(n_features)

    contrib = booster.predict(features, pred_contrib=True)
    contrib = np.asarray(contrib).reshape(-1, n_classes, n_features + 1)
    per_feat = contrib[0, pred_idx, :-1]  # bias 제외

    # 물리 그룹별 기여도 합산
    sums: dict[str, float] = {}
    for g, v in zip(_GROUP_IDX, per_feat):
        sums[g] = sums.get(g, 0.0) + float(v)

    ranked = sorted(sums.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    out = []
    for key, val in ranked:
        out.append({
            "factor": GROUP_LABEL.get(key, key),
            "impact": f"{val:+.2f}",
            "effect": "진단을 뒷받침" if val >= 0 else "진단과 반대 방향",
        })
    return out
