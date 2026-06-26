"""
evaluate_enhanced.py
--------------------
저장된 강화 모델(robot_pdm_enhanced.txt)을 독립적으로 불러와
공식 Validation set으로 실제 측정 + 혼동행렬 + 주요 오분류를 출력한다.
"""
import numpy as np, lightgbm as lgb, json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

DATA = '../data/processed'

def feat(X, S):
    N = X.shape[0]
    eng = np.hstack([X.reshape(N, -1), np.mean(X, 1), np.std(X, 1),
        np.mean(X[:, -10:, :], 1) - np.mean(X[:, :10, :], 1),
        np.abs(np.fft.rfft(X, axis=1))[:, :15, :].reshape(N, -1)])
    return np.hstack([eng, S]).astype(np.float32)

def main():
    booster = lgb.Booster(model_file=f'{DATA}/robot_pdm_enhanced.txt')
    mm = json.load(open(f'{DATA}/robot_pdm_enhanced_meta.json', encoding='utf-8'))
    names = mm['class_names']; classes = np.array(mm['classes'])
    va = np.load(f'{DATA}/enhanced_val.npz')
    X = feat(va['X'], va['S']); y = va['y']
    yp = classes[np.argmax(booster.predict(X), axis=1)]

    acc = accuracy_score(y, yp); f1m = f1_score(y, yp, average='macro')
    base = max(np.mean(y == c) for c in np.unique(y))
    print(f'공식 Validation {len(y)}건  정확도 {acc*100:.2f}%  macro-F1 {f1m:.4f}  (기준선 {base*100:.2f}%)\n')
    print(classification_report(y, yp, digits=3, target_names=names))

    print('\n=== 혼동행렬 (행=실제, 열=예측) ===')
    cm = confusion_matrix(y, yp, labels=list(range(len(names))))
    print('실제\\예측 ' + ' '.join(f'{i:>5}' for i in range(len(names))))
    for i in range(len(names)):
        print(f'{i:>2} {names[i][:8]:<8} ' + ' '.join(f'{cm[i,j]:>5}' for j in range(len(names))))
    print('\n클래스 인덱스:', {i: n for i, n in enumerate(names)})

    print('\n=== 주요 오분류 Top5 ===')
    mis = [(i, j, cm[i, j]) for i in range(len(names)) for j in range(len(names)) if i != j]
    for i, j, c in sorted(mis, key=lambda t: -t[2])[:5]:
        print(f'  {names[i]} -> {names[j]}: {c}건')

if __name__ == '__main__':
    main()
