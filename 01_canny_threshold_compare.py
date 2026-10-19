import cv2 as cv
import os

base=os.path.dirname(os.path.abspath(__file__))
out_dir=os.path.join(base,'output')
os.makedirs(out_dir,exist_ok=True)

img=cv.imread(os.path.join(base,'docs','images','structure_1.jpg'))	# 영상 읽기
if img is None:
    raise FileNotFoundError('docs/images/structure_1.jpg를 읽을 수 없습니다.')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

canny1=cv.Canny(gray,50,150)	# Tlow=50, Thigh=150으로 설정
canny2=cv.Canny(gray,100,200)	# Tlow=100, Thigh=200으로 설정

# 결과를 파일로 저장
cv.imwrite(os.path.join(out_dir,'01_original_gray_structure_1.png'),gray)
cv.imwrite(os.path.join(out_dir,'01_canny1_50_150_structure_1.png'),canny1)
cv.imwrite(os.path.join(out_dir,'01_canny2_100_200_structure_1.png'),canny2)
print('저장 완료:',out_dir)
