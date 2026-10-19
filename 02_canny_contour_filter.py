import cv2 as cv
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base, 'output')
os.makedirs(out_dir, exist_ok=True)

img = cv.imread(os.path.join(base, 'docs', 'images', 'structure_1.jpg'))
if img is None:
    raise FileNotFoundError('docs/images/structure_1.jpg를 읽을 수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 100, 200)
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.imwrite(os.path.join(out_dir, '02_canny_structure_1.png'), canny)

for min_len in [100, 300, 500]:    # 비교할 길이 기준값
    lcontour = [c for c in contour if c.shape[0] > min_len]
    result = img.copy()            # 매번 원본 복사본에 그림
    cv.drawContours(result, lcontour, -1, (0, 255, 0), 3)
    name = f'02_contours_len{min_len}_structure_1.png'
    cv.imwrite(os.path.join(out_dir, name), result)
    print(f'기준 {min_len}: 윤곽 {len(lcontour)}개 -> {name}')

print('저장 완료:', out_dir)
