import cv2
import numpy as np
import supportVACT as sp
import solveImg as si
path = r'C:\review_OMR\images\dgnl.jpg'
widthImg = 1000
heightImg = 1400
questions = 120
choices = 4
stack = questions //5
finalAns = (
    [0] * 10 +  # 10 câu A
    [1] * 10 +  # 10 câu B
    [2] * 10 +  # 10 câu C
    [3] * 10 +  # 10 câu D
    [0] * 10 +   # 10 câu A
    [1] * 10 +  # 10 câu A
    [0] * 10 +  # 10 câu A
    [1] * 10 +  # 10 câu B
    [2] * 10 +  # 10 câu C
    [3] * 10 +  # 10 câu D
    [0] * 10 +   # 10 câu A
    [1] * 10    # 10 câu A
)
img = cv2.imread(path)
img = cv2.resize(img,(widthImg,heightImg))
imgAns = img.copy()
imgFinal = img.copy()
cv2.namedWindow("originally", cv2.WINDOW_NORMAL)
#Gray
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
# Canny
imgCanny = cv2.Canny(imgBlur, 50, 150)
#cv2.imshow("Canny",imgCanny)
#contour
contours,h = cv2.findContours(imgCanny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#loc ra contours tu giac area > 50
contours = sp.rectContour(contours)
#cv2.drawContours(img, contours, -1, (0,255,0), 2)
img,pointContour = si.takeImageAnswer(img,contours,[60,100,800,850],[550,650,1200,1300])
per_x1 = int(pointContour[0][0]) - 12
per_y1 = int(pointContour[0][1]) + 26
per_x2 = 965
per_y2 = int(pointContour[2][1])
imgPer = imgAns[per_y1:per_y2,per_x1:per_x2]
imgCVT = cv2.cvtColor(imgPer,cv2.COLOR_BGR2GRAY)
imgThresh = cv2.threshold(imgCVT, 135, 255, cv2.THRESH_BINARY_INV)[1]
#cv2.imshow("originally",imgThresh)
boxes = sp.splitImg(imgThresh)
x = 1
#cv2.imshow('boxpiece', boxes[x*4])
#cv2.imshow('boxpiece1', boxes[x*4+1])
#cv2.imshow('boxpiece2', boxes[x*4+2])
#cv2.imshow('boxpiece3', boxes[x*4+3])
ans = sp.splitAns(boxes,12,4)
pixelVals = np.zeros((120, choices))
countC = 0
countR = 0
#count pixel
for image in ans:
    totalPixels = cv2.countNonZero(image)
    pixelVals[countR][countC] = totalPixels
    countC += 1
    if(countC == choices):
        countR += 1
        countC = 0
#print(pixelVals)
#index
myIndex = []
for x in range(len(finalAns)):

    arr = pixelVals[x]
    validIndex = np.where(arr > 200)[0]
    if len(validIndex) == 0:
        myIndex.append(-1)
    elif len(validIndex) > 1:
        myIndex.append(-2)
    else:
        myIndex.append(validIndex[0])
print(myIndex)
#grading
grading = []

for x in range(0, len(finalAns)):
    if( myIndex[x] == finalAns[x] ):
        grading.append(1)
    else:
        grading.append(0)
print(grading)
rawBoxes = sp.splitImg(imgPer)
for x in range(len(finalAns)//12):
    stackAns = finalAns[x*12:x*12+12]
    stackIndex = myIndex[x*12:x*12+12]
    stackGrading = grading[x*12:x*12+12]
    rawBoxes[x] = sp.showAnswer(rawBoxes[x], stackIndex, stackAns, stackGrading, 12, 4)
imgRawFinal = sp.restoreImg(imgPer,rawBoxes,stack)
imgAns[per_y1:per_y2,per_x1:per_x2] = imgRawFinal
cv2.imshow("originally",imgAns)
if cv2.waitKey(0) == ord('x'):
    cv2.destroyAllWindows()