"""
train_enhanced.py
-----------------
강화 데이터셋(버려졌던 필드 포함 + 공식 Training/Validation split)으로 학습/평가.
- 동적센서 7개: flatten + mean/std/trend/FFT
- 정적/누적/맥락 9개(distance, batteryCycleCount, degree계열, emergencyStop, crowd, deviceType, mainState 등) 부착
- 평가: 데이터셋 공식 Validation set (학습에 안 쓰임 -> 누수 0, 정직)
출력: robot_pdm_enhanced.txt + robot_pdm_enhanced_meta.json
"""
import numpy as np, lightgbm as lgb, json, os, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

DATA = '../data/processed'

# 모델이 쓰는 동적센서: 절대좌표 x,y(index 2,3)는 제외(사이트별 좌표 암기 방지).
# DYN 원순서=[batteryLevel,speed,x,y,degree,collision,obstacle]
MODEL_DYN_IDX = [0, 1, 4, 5, 6]  # batteryLevel, speed, degree, collision, obstacle

def feat(X, S):
    X = X[:, :, MODEL_DYN_IDX]
    N = X.shape[0]
    eng = np.hstack([
        X.reshape(N, -1), np.mean(X, 1), np.std(X, 1),
        np.mean(X[:, -10:, :], 1) - np.mean(X[:, :10, :], 1),
        np.abs(np.fft.rfft(X, axis=1))[:, :15, :].reshape(N, -1),
    ])
    return np.hstack([eng, S]).astype(np.float32)

def main():
    t0 = time.time()
    tr = np.load(f'{DATA}/enhanced_train.npz'); va = np.load(f'{DATA}/enhanced_val.npz')
    meta = json.load(open(f'{DATA}/enhanced_meta.json', encoding='utf-8'))
    err_map = meta['err_map']; inv = {v: k for k, v in err_map.items()}
    names = [inv[i] for i in range(len(inv))]
    normal_cls = err_map['정상']
    print(f'train {tr["X"].shape}  val {va["X"].shape}  | 클래스 {len(names)}개')

    Xtr_raw, Str, ytr = tr['X'], tr['S'], tr['y']
    Xva, Sva, yva = feat(va['X'], va['S']), None, va['y']
    Xva = feat(va['X'], va['S'])

    # 학습부분 언더샘플링(정상=에러총합 2배) — 희귀고장 학습 보호
    err = np.where(ytr != normal_cls)[0]; norm = np.where(ytr == normal_cls)[0]
    rng = np.random.RandomState(42)
    keep = rng.choice(norm, size=min(len(err) * 3, len(norm)), replace=False)  # 3x가 acc/F1 최적
    sel = np.concatenate([err, keep]); rng.shuffle(sel)
    Xtr = feat(Xtr_raw[sel], Str[sel]); ytr_s = ytr[sel]
    print(f'학습표본 {len(sel)} (에러 {len(err)} + 정상 {len(keep)})')

    Xt, Xv, yt, yv = train_test_split(Xtr, ytr_s, test_size=0.12, random_state=42, stratify=ytr_s)
    model = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=63,
        max_depth=8, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1)
    model.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(40, verbose=False)])

    yp = model.predict(Xva)
    acc = accuracy_score(yva, yp); f1m = f1_score(yva, yp, average='macro')
    base = max(np.mean(yva == c) for c in np.unique(yva))
    print('\n' + '=' * 56)
    print(f'공식 Validation  정확도 {acc*100:.2f}%   macro-F1 {f1m:.4f}')
    print(f'(무조건 정상 기준선 {base*100:.2f}%, best_iter={model.best_iteration_})')
    print('=' * 56)
    print(classification_report(yva, yp, digits=3, target_names=names))

    # 피처 중요도 top12
    mdyn = [meta['dyn'][i] for i in MODEL_DYN_IDX]
    imp = model.feature_importances_
    eng_names = ([f'{s}_t{t}' for t in range(30) for s in mdyn]
                 + [f'{s}_mean' for s in mdyn] + [f'{s}_std' for s in mdyn]
                 + [f'{s}_trend' for s in mdyn]
                 + [f'fft{k}_{s}' for k in range(15) for s in mdyn]
                 + meta['stat'])
    print('\n=== 피처 중요도 Top12 ===')
    for i in np.argsort(imp)[::-1][:12]:
        nm = eng_names[i] if i < len(eng_names) else f'f{i}'
        print(f'  {nm:<18} {imp[i]}')

    model.booster_.save_model(f'{DATA}/robot_pdm_enhanced.txt')
    json.dump({'classes': model.classes_.tolist(), 'class_names': names,
               'n_features': Xtr.shape[1], 'val_acc': round(acc, 4),
               'val_macro_f1': round(f1m, 4), 'best_iteration': int(model.best_iteration_),
               'dyn': meta['dyn'], 'model_dyn_idx': MODEL_DYN_IDX, 'stat': meta['stat'], 'err_map': err_map},
              open(f'{DATA}/robot_pdm_enhanced_meta.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    sz = os.path.getsize(f'{DATA}/robot_pdm_enhanced.txt') / 1e6
    print(f'\n저장 robot_pdm_enhanced.txt ({sz:.2f} MB) | 소요 {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
