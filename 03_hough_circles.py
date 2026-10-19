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

img = cv.imread(find_image('coin.png'))
if img is None:
    raise FileNotFoundError('coin.png를 읽을 수 없습니다.')

# 해상도에 따라 반지름 값이 달라지므로, 가로 1200px로 줄여서 같은 기준으로 비교
W = 1200
img = cv.resize(img, (W, int(img.shape[0] * W / img.shape[1])))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.medianBlur(gray, 5)        # 노이즈 제거 (HoughCircles는 노이즈에 민감)

cv.imwrite(os.path.join(out_dir, '03_original_coin.png'), img)
cv.imwrite(os.path.join(out_dir, '03_blur_coin.png'), blur)

# param1: 내부 Canny의 높은 임계값 (낮은 임계값은 그 절반)
# param2: 원으로 인정하는 누적 투표 임계값 (낮을수록 많이 검출, 오검출 증가)
for param2 in [20, 30, 40, 50]:
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, 40,
                              param1=100, param2=param2,
                              minRadius=30, maxRadius=75)
    result = img.copy()
    count = 0
    if circles is not None:                 # 원이 하나도 없으면 None
        for x, y, r in circles[0]:
            cv.circle(result, (int(x), int(y)), int(r), (255, 0, 0), 2)   # 원 (파랑, BGR)
            cv.circle(result, (int(x), int(y)), 2, (0, 0, 255), 3)        # 중심점 (빨강)
        count = circles.shape[1]
    name = f'03_circles_param2_{param2}_coin.png'
    cv.imwrite(os.path.join(out_dir, name), result)
    print(f'param2={param2}: 원 {count}개 -> {name}')

print('저장 완료:', out_dir)