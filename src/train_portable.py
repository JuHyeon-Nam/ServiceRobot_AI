"""
train_portable.py
------------------
작고 휴대 가능한(어디서든 CPU로 추론되는) 서비스 로봇 PdM 모델을 만든다.

핵심 개선점:
  1. 올바른 방법론: 원본(불균형) 데이터에서 먼저 holdout을 떼고,
     '학습 부분만' 언더샘플링한다. -> 정직한 실전 성능 측정.
  2. 컴팩트한 LightGBM(num_leaves=63, early stopping) -> 217MB -> 수 MB.
  3. native text 포맷(.txt)으로 저장 -> 언어/플랫폼 독립, 초경량.
  4. 추론에 필요한 피처 엔지니어링 함수를 같은 파일에 둬서 서빙과 일치시킨다.
"""
import numpy as np
import lightgbm as lgb
import json, os, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

DATA = '../data/processed'
SEQ_LEN, N_SENSORS = 30, 7

# ---- 서빙과 100% 동일해야 하는 피처 엔지니어링 ----
def create_features(X_raw):
    """X_raw: (N, 30, 7) -> (N, 336)"""
    N = X_raw.shape[0]
    X_flat   = X_raw.reshape(N, -1)                       # 210
    t_mean   = np.mean(X_raw, axis=1)                     # 7
    t_std    = np.std(X_raw, axis=1)                      # 7
    trend    = np.mean(X_raw[:, -10:, :], axis=1) - np.mean(X_raw[:, :10, :], axis=1)  # 7
    fft_half = np.abs(np.fft.rfft(X_raw, axis=1))[:, :15, :].reshape(N, -1)            # 105
    return np.hstack([X_flat, t_mean, t_std, trend, fft_half]).astype(np.float32)

def main():
    t0 = time.time()
    print('데이터 로딩(원본 불균형 전체)...')
    X = np.load(f'{DATA}/X_train.npy', mmap_mode='r')      # (1.05M, 30, 7)
    y = np.load(f'{DATA}/y_train_multi.npy')
    print(f'  X={X.shape}, y={y.shape}, 클래스 분포 정상비율={np.mean(y==9)*100:.1f}%')

    # 1) 원본에서 holdout 분리 (실전 분포 그대로 유지) -------------
    idx = np.arange(len(y))
    tr_idx, ho_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)

    # 2) 학습 부분만 언더샘플링 (정상 9번을 에러총합의 2배로) -------
    y_tr = y[tr_idx]
    err_idx = tr_idx[y_tr != 9]
    norm_idx = tr_idx[y_tr == 9]
    rng = np.random.RandomState(42)
    keep_norm = rng.choice(norm_idx, size=min(len(err_idx) * 2, len(norm_idx)), replace=False)
    train_idx = np.concatenate([err_idx, keep_norm])
    rng.shuffle(train_idx)
    print(f'  학습표본={len(train_idx)} (에러 {len(err_idx)} + 정상 {len(keep_norm)}), holdout={len(ho_idx)}')

    # 3) 피처 생성 (인덱스로 뽑아서 메모리 절약) -------------------
    print('피처 엔지니어링...')
    Xtr = create_features(np.asarray(X[np.sort(train_idx)]))
    ytr = y[np.sort(train_idx)]
    Xho = create_features(np.asarray(X[np.sort(ho_idx)]))
    yho = y[np.sort(ho_idx)]

    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=42, stratify=ytr)

    # 4) 컴팩트 모델 ---------------------------------------------
    print('학습(컴팩트 LightGBM + early stopping)...')
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        max_depth=8, min_child_samples=40, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    print(f'  best_iteration={model.best_iteration_}')

    # 5) 정직한 평가 (실전 불균형 holdout) ------------------------
    yp = model.predict(Xho)
    acc = accuracy_score(yho, yp)
    f1m = f1_score(yho, yp, average='macro')
    print('\n' + '=' * 50)
    print(f'실전(불균형) holdout  정확도 {acc*100:.2f}%   macro-F1 {f1m:.4f}')
    print(f'(참고: 무조건 정상=80.5% 기준선)')
    print('=' * 50)
    print(classification_report(yho, yp, digits=3))

    # 6) 휴대용 저장 ---------------------------------------------
    os.makedirs(f'{DATA}', exist_ok=True)
    txt_path = f'{DATA}/robot_pdm_portable.txt'
    model.booster_.save_model(txt_path)
    meta = {
        'n_features': Xtr.shape[1], 'classes': model.classes_.tolist(),
        'best_iteration': int(model.best_iteration_),
        'holdout_acc': round(acc, 4), 'holdout_macro_f1': round(f1m, 4),
        'feature_order': 'flat210 + mean7 + std7 + trend7 + rfft15x7=105',
    }
    json.dump(meta, open(f'{DATA}/robot_pdm_meta.json', 'w'), ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(txt_path) / 1e6
    print(f'\n저장: {txt_path}  ({size_mb:.2f} MB)  | meta: robot_pdm_meta.json')
    print(f'총 소요 {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
