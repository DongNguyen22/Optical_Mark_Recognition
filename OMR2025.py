import cv2
import MDD as md
import support2020 as sp
import solveImg as si
import numpy as np
path = r'C:\review_OMR\efg.jpg'
widthImg,heightImg = 500,700
choices = 4
img = cv2.imread(path)
img = cv2.resize(img,(widthImg,heightImg))
imgAns = img.copy()
imgPart_1_original = img.copy()
imgPart_2_original = img.copy()
imgPart_3_original = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Gray
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
#threshold
imgThresh = cv2.threshold(imgBlur,135,255,cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("thresh",imgThresh)
# Canny
imgCanny = cv2.Canny(imgBlur, 50, 150)
#cv2.imshow("Canny",imgCanny)
#contour
contours,_ = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#loc ra contours tu giac area > 50
#part1
imgPart_1_original,pointContour_1 = si.takeImageAnswer(imgPart_1_original,contours,[2,20,480,500],[200,235,380,400])
imgPart_2_original,pointContour_2 = si.takeImageAnswer(imgPart_2_original,contours,[2,20,480,500],[370,400,490,510])
imgPart_3_original,pointContour_3 = si.takeImageAnswer(imgPart_3_original,contours,[2,20,480,500],[475,500,670,700])
#part_1
pointContour_1 = sp.reorder(pointContour_1)

p1_x1 = pointContour_1[0][0][0] + 8
p1_y1 = pointContour_1[0][0][1] + 40

p1_x2 = pointContour_1[3][0][0]
p1_y2 = pointContour_1[3][0][1] - 8
#part_2
pointContour_2 = sp.reorder(pointContour_2)

p2_x1 = pointContour_2[0][0][0] + 8
p2_y1 = pointContour_2[0][0][1] + 40

p2_x2 = pointContour_2[3][0][0]
p2_y2 = pointContour_2[3][0][1] - 8
#part_3
pointContour_3 = sp.reorder(pointContour_3)

p3_x1 = pointContour_3[0][0][0] + 8
p3_y1 = pointContour_3[0][0][1] + 35

p3_x2 = pointContour_3[3][0][0]
p3_y2 = pointContour_3[3][0][1] - 8

imgPart_1 = img[p1_y1:p1_y2, p1_x1:p1_x2]
imgPart_2 = img[p2_y1:p2_y2, p2_x1:p2_x2]
imgPart_3 = img[p3_y1:p3_y2, p3_x1:p3_x2]
#cv2.imshow("part1",imgPart_1)
#cv2.imshow("part2",imgPart_2)
#cv2.imshow("part1",imgPart_1)
boxes_1 = sp.splitImg(imgPart_1,1,4)
kkk = sp.splitImg(imgPart_2,1,4)
boxes_3 = sp.splitImg(imgPart_3,1,6)
boxes_2= []
for image in kkk:
    h,w = image.shape[:2]
    middle = int(w//2)
    piece_1 = image[0:h,0:middle-1]
    piece_2 = image[0:h,middle:w]
    boxes_2.append(piece_1)
    boxes_2.append(piece_2)
ans_1 = sp.splitAns(boxes_1,10,4)
h,w = boxes_3[0].shape[:2]
boxes_3[0] = cv2.resize(boxes_3[0],(100,h))
ans_part3 = sp.splitAns2025(boxes_3[0],12,4)
cv2.imshow("box_3",boxes_3[0])
cv2.imshow("p3_0",ans_part3[0][0])
cv2.imshow("p3_1",ans_part3[0][1])
#pixelValues_1 = sp.countPixel(ans_1,40,4)
#myIndex_1 = sp.countIndex(pixelValues_1,40)


if cv2.waitKey(0) == ord('x'):
    cv2.destroyAllWindows()