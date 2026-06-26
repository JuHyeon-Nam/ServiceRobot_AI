"""
evaluate_portable.py
--------------------
저장된 휴대용 모델(robot_pdm_portable.txt)을 '독립적으로' 불러와
실전 불균형 holdout으로 실제 측정 테스트한다.
-> 모델 파일 하나만 있으면 어디서든 평가/추론된다는 것을 증명.
"""
import numpy as np
import lightgbm as lgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

DATA = '../data/processed'

def create_features(X_raw):
    N = X_raw.shape[0]
    return np.hstack([
        X_raw.reshape(N, -1), np.mean(X_raw, 1), np.std(X_raw, 1),
        np.mean(X_raw[:, -10:, :], 1) - np.mean(X_raw[:, :10, :], 1),
        np.abs(np.fft.rfft(X_raw, axis=1))[:, :15, :].reshape(N, -1),
    ]).astype(np.float32)

NAMES = {0:'구동부', 1:'배터리', 2:'통신과열', 3:'센서', 4:'바퀴슬립',
         5:'범퍼', 6:'모터(희귀)', 7:'라이다(희귀)', 8:'엔코더', 9:'정상'}

def main():
    booster = lgb.Booster(model_file=f'{DATA}/robot_pdm_portable.txt')
    meta = json.load(open(f'{DATA}/robot_pdm_meta.json', encoding='utf-8'))
    classes = meta['classes']
    print(f"모델 로드: {meta['n_features']}피처, 학습시 best_iter={meta['best_iteration']}")

    # 학습과 동일한 시드로 holdout 재현 (같은 데이터로 측정)
    X = np.load(f'{DATA}/X_train.npy', mmap_mode='r')
    y = np.load(f'{DATA}/y_train_multi.npy')
    idx = np.arange(len(y))
    _, ho_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
    ho_idx = np.sort(ho_idx)

    Xho = create_features(np.asarray(X[ho_idx]))
    yho = y[ho_idx]

    probs = booster.predict(Xho)
    yp = np.array(classes)[np.argmax(probs, axis=1)]

    acc = accuracy_score(yho, yp)
    f1m = f1_score(yho, yp, average='macro')
    print(f'\n실전(불균형) holdout {len(yho)}건  정확도 {acc*100:.2f}%  macro-F1 {f1m:.4f}\n')
    print(classification_report(yho, yp, digits=3,
          target_names=[f'{i}:{NAMES[i]}' for i in range(10)]))

    print('\n=== 혼동행렬 (행=실제, 열=예측) ===')
    cm = confusion_matrix(yho, yp, labels=list(range(10)))
    hdr = '실제\\예측 ' + ' '.join(f'{i:>6}' for i in range(10))
    print(hdr)
    for i in range(10):
        print(f'{i:>2}:{NAMES[i]:<7} ' + ' '.join(f'{cm[i,j]:>6}' for j in range(10)))

    # 가장 자주 틀리는 오분류 top5
    print('\n=== 주요 오분류 Top5 (실제->예측, 건수) ===')
    mis = [(i, j, cm[i, j]) for i in range(10) for j in range(10) if i != j]
    for i, j, c in sorted(mis, key=lambda t: -t[2])[:5]:
        print(f'  {NAMES[i]}({i}) -> {NAMES[j]}({j}): {c}건')

if __name__ == '__main__':
    main()
