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

# 해상도에 따라 blockSize 효과가 달라지므로, 가로 1500px로 줄여서 같은 기준으로 비교
W = 1500
img = cv.resize(img, (W, int(img.shape[0] * W / img.shape[1])))
H = img.shape[0]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 글자가 작아서 비교하기 좋은 확대 구간 (그림자가 있는 쪽 표의 일부)
def detail(im):
    return im[int(H * 0.725):int(H * 0.875), int(W * 0.467):int(W * 0.933)]

# 1) 방법 비교: 고정 임계값 / Otsu / Adaptive(평균, 가우시안)
fixed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)[1]
otsu_t, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
a_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 10)
a_gauss = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 10)
print(f'Otsu가 고른 임계값: {otsu_t:.0f}')

results = {'gray': gray, 'fixed127': fixed, 'otsu': otsu,
           'adaptive_mean': a_mean, 'adaptive_gaussian': a_gauss}
for name, im in results.items():
    cv.imwrite(os.path.join(out_dir, f'05_{name}_document.png'), im)
    cv.imwrite(os.path.join(out_dir, f'05_detail_{name}_document.png'), detail(im))
    if name != 'gray':
        print(f'{name}: 검은 픽셀 {100 * (im == 0).mean():.1f}%')

# 2) 파라미터 비교
# blockSize: 임계값을 계산하는 주변 영역의 크기 (홀수)
# C        : 주변 평균에서 빼는 상수 (클수록 배경이 깨끗해지지만 얇은 글자가 사라짐)
for block in [11, 31, 51, 101]:
    im = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, 10)
    cv.imwrite(os.path.join(out_dir, f'05_detail_block{block}_C10_document.png'), detail(im))
for c in [2, 20]:
    im = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, c)
    cv.imwrite(os.path.join(out_dir, f'05_detail_block31_C{c}_document.png'), detail(im))

print('저장 완료:', out_dir)