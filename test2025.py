import cv2
import MDD as md
import support2025 as sp
import solveImg as si
import MDD2025 as mdd
import numpy as np
path = r'C:\review_OMR\images\nghieng.jpg'
final_ans1 = (
    [0] * 10 +  # 10 câu A
    [1] * 10 +  # 10 câu B
    [2] * 10 +  # 10 câu C
    [3] * 10   # 10 câu D
)
# Part 2: 8 câu đúng sai
# Đúng = 1, Sai = 0

final_ans2 = [
    [1,0,1,0],
    [0,0,1,1],
    [1,1,0,0],
    [0,1,0,1],
    [1,1,1,0],
    [0,0,0,1],
    [1,0,0,1],
    [1,1,0,1]
]
final_ans3 = sp.change_to_ans_part3([ '1,34','10', '12,5', '1,83', '12','-1,5'])
choices = 4
img = cv2.imread(path)
widthImg,heightImg = 1000,1400
img = cv2.resize(img, (widthImg,heightImg))
imgAns = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Gray
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
#threshold
imgThresh = cv2.threshold(imgBlur,135,255,cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("thresh",imgThresh)
#cv2.namedWindow("original", cv2.WINDOW_NORMAL)
#cv2.imshow("original",img)
# Canny
imgCanny = cv2.Canny(imgBlur, 50, 150)
#cv2.imshow("Canny",imgCanny)
#contour
contours,h = cv2.findContours(imgCanny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#loc ra contours tu giac area > 50
contours = sp.rectContour(contours)
#cv2.drawContours(img, contours, -1, (0,255,0), 2)
#part1
imgCopy = img.copy()
imgCopy ,pointContour = si.takeImageAnswer(imgCopy,contours,[50,150,930,970],[400,450,1200,1300])
#cv2.namedWindow("first",cv2.WINDOW_NORMAL)
#cv2.imshow("part1",imgCopy)
p1_x1 = int(pointContour[0][0])
p1_y1 = int(pointContour[0][1])

p1_x2 = int(pointContour[3][0]) + 50
p1_y2 = int(pointContour[3][1]) + 50
imgPart = imgCopy[p1_y1:p1_y2,p1_x1:p1_x2]
#cv2.imshow("first",imgPart)

pt1 = np.float32([pointContour[0], pointContour[1], pointContour[2], pointContour[3]])
pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgWarpColored = cv2.warpPerspective(imgAns, matrix, (widthImg, heightImg))
height_warp,width_warp = imgWarpColored.shape[:2]
#Gray
imgGray_1 = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
# Blur
imgBlur_1 = cv2.GaussianBlur(imgGray_1, (5,5), 1)
# Canny
imgCanny_1 = cv2.Canny(imgBlur_1, 50, 150)
#contour
contours_1,h_1 = cv2.findContours(imgCanny_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#loc ra contours tu giac area > 50
contours_1 = sp.rectContour(contours_1)
#cv2.namedWindow("blank", cv2.WINDOW_NORMAL)
#cv2.imshow('blank', imgWarpColored)
imgPart_1_original = imgWarpColored.copy()
imgPart_2_original = imgWarpColored.copy()
imgPart_3_original = imgWarpColored.copy()
imgPart_1_original,pointContour_1 = si.takeImageAfterWarp(imgPart_1_original,contours_1,[0,20,0,0],[0,50,500,550],[0,1,0,1])
imgPart_2_original,pointContour_2 = si.takeImageAfterWarp(imgPart_2_original,contours_1,[0,20,0,0],[450,520,800,840],[0,1,0,1])
p1_x1,p1_y1 = int(pointContour_1[3][0]), int(pointContour_1[0][1]) +10
p1_y2 =int(pointContour_1[2][1]) +10
part_1 = imgWarpColored[p1_y1:p1_y2,p1_x1:int(width_warp)]
#cv2.imshow("part_1",part_1)
#cv2.namedWindow("p1", #cv2.WINDOW_NORMAL)
#cv2.imshow("p1",imgPart_1_original)

p2_x1,p2_y1 = int(pointContour_2[3][0]) +5 , int(pointContour_2[0][1]) + 20
p2_y2 =int(pointContour_2[2][1]) - 20
part_2 = imgWarpColored[p2_y1:p2_y2,p2_x1:int(width_warp)]
cv2.namedWindow("part_2", cv2.WINDOW_NORMAL)
#cv2.imshow("p2",imgPart_2_original)
cv2.imshow("part_2",part_2)

if cv2.waitKey(0) == ord('x'):
    cv2.destroyAllWindows()