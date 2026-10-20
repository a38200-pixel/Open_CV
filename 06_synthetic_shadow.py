import cv2 as cv
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base, 'output')
os.makedirs(out_dir, exist_ok=True)

# 이미지 폴더: docs/images 를 먼저 찾고, 없으면 images 를 찾음
def find_image(name):
    for folder in (os.path.join(base, 'docs', 'images'), os.path.join(base, 'images')):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{name}을(를) docs/images 또는 images 폴더에서 찾을 수 없습니다.')

img = cv.imread(find_image('document.jpg'))
if img is None:
    raise FileNotFoundError('document.jpg를 읽을 수 없습니다.')

W = 1500
img = cv.resize(img, (W, int(img.shape[0] * W / img.shape[1])))
H = img.shape[0]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 평가 영역: 표 안쪽만 사용 (배경, 컬러 머리글, 일러스트 제외)
y0, y1, x0, x1 = int(H * 0.29), int(H * 0.85), int(W * 0.133), int(W * 0.827)
roi = (slice(y0, y1), slice(x0, x1))

# 기준(reference): 그림자를 더하기 전 원본의 Otsu 결과 (정답이 아니라 비교용 기준)
ref = cv.threshold(gray[roi], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

def f1_text(pred, ref):
    """글자(검은 픽셀)가 기준과 얼마나 일치하는지 F1 점수로 계산"""
    a, b = (pred == 0), (ref == 0)
    tp = (a & b).sum()
    fp = (a & ~b).sum()
    fn = (~a & b).sum()
    return 2 * tp / (2 * tp + fp + fn)

def binarize(im, block=31, c=10):
    return {
        'fixed127': cv.threshold(im, 127, 255, cv.THRESH_BINARY)[1],
        'otsu': cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1],
        'adaptive_mean': cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block, c),
        'adaptive_gaussian': cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, c),
    }

# 왼쪽은 그대로, 오른쪽으로 갈수록 alpha 비율만큼 어두워지는 가짜 그림자
ramp = np.linspace(0, 1, W)[None, :].repeat(H, 0)

def add_shadow(im, alpha):
    return (im * (1 - alpha * ramp)).clip(0, 255).astype(np.uint8)

# 1) 그림자 강도별 점수
names = ['fixed127', 'otsu', 'adaptive_mean', 'adaptive_gaussian']
rows = ['alpha,' + ','.join(names)]
print('그림자 강도(alpha)별 F1 (1에 가까울수록 기준과 일치)')
print('alpha  ' + '  '.join(f'{n:>17}' for n in names))
for alpha in [0, 0.2, 0.35, 0.5, 0.65]:
    shadowed = add_shadow(gray, alpha)
    res = binarize(shadowed)
    scores = [f1_text(res[n][roi], ref) for n in names]
    print(f'{alpha:<6} ' + '  '.join(f'{s:>17.3f}' for s in scores))
    rows.append(f'{alpha},' + ','.join(f'{s:.4f}' for s in scores))
    if alpha in (0.5,):
        cv.imwrite(os.path.join(out_dir, '06_shadowed_gray_alpha0.5.png'), shadowed)
        for n in names:
            cv.imwrite(os.path.join(out_dir, f'06_alpha0.5_{n}.png'), res[n])
with open(os.path.join(out_dir, '06_synthetic_shadow_scores.csv'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(rows))

# 2) alpha=0.5 에서 blockSize 비교 (adaptive gaussian, C=10)
shadowed = add_shadow(gray, 0.5)
print('\nblockSize별 F1 (alpha=0.5, adaptive gaussian, C=10)')
for block in [11, 31, 51, 101, 201]:
    res = cv.adaptiveThreshold(shadowed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, 10)
    print(f'block={block:<4} F1={f1_text(res[roi], ref):.3f}')

print('저장 완료:', out_dir)