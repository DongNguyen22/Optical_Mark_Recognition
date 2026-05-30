import cv2
import numpy as np
import support2020 as sp
import solveImg as si
# import MDD as mdd
path = r'C:\Code VSCode\OMR\Optical_Mark_Recognition\images\2020.jpg'
widthImg = 1000
heightImg = 1400
questions = 50
choices = 4
stack = questions//5
MDD = []
MD = []
finalAns = (
    [0] * 10 +  # 10 câu A
    [1] * 10 +  # 10 câu B
    [2] * 10 +  # 10 câu C
    [3] * 10 +  # 10 câu D
    [0] * 10    # 10 câu A
)
img = cv2.imread(path)
img = cv2.resize(img,(widthImg,heightImg))
imgAns = img.copy()
imgFinal = img.copy()
#cv2.imshow("originally",img)
#Gray
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
# Canny
imgCanny = cv2.Canny(imgBlur, 50, 150)
#cv2.imshow("Canny",imgCanny)
#contour
contours,h = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#loc ra contours tu giac area > 50
contours = sp.rectContour(contours)
#cv2.drawContours(img, contours, -1, (0,255,0), 2)
img,pointContour = si.takeImageAnswer(img,contours,[50,150,900,1000],[380,450,1250,1300])
#print(pointContour)
#cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
#cv2.imshow("contours",img)
pt1 = np.float32([pointContour[0], pointContour[1], pointContour[2], pointContour[3]])
pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgWarpColored = cv2.warpPerspective(imgAns, matrix, (widthImg, heightImg))
#cv2.namedWindow("first", cv2.WINDOW_NORMAL)
#cv2.imshow('blank', imgWarpColored)

per_x1,per_x2,per_y1,per_y2 = 8,982,72,1375
imgPer = imgWarpColored[per_y1:per_y2,per_x1:per_x2]
#cv2.imshow("first",imgPer)
imgRevert = imgPer.copy()
imgCvt = cv2.cvtColor(imgPer, cv2.COLOR_BGR2GRAY)
imgThresh = cv2.threshold(imgCvt, 135, 255, cv2.THRESH_BINARY_INV)[1]
boxes = sp.splitImg(imgThresh)
#x = 0
#cv2.imshow('boxpiece', boxes[x*4])
#cv2.imshow('boxpiece1', boxes[x*4+1])
#cv2.imshow('boxpiece2', boxes[x*4+2])
#cv2.imshow('boxpiece3', boxes[x*4+3])
#cv2.imshow("final",img)

#answer
ans = sp.splitAns(boxes,5,4)
x = 0
#cv2.imshow("ans1",ans[x*4])
#cv2.imshow("ans2",ans[x*4+1])
#cv2.imshow("ans3",ans[x*4+2])
#cv2.imshow("ans4",ans[x*4+3])
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
for x in range(questions):

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

for x in range(0, questions):
    if( myIndex[x] == finalAns[x] ):
        grading.append(1)
    else:
        grading.append(0)
imgRawRevert = np.zeros_like(imgRevert)
rawBoxes = sp.splitImg(imgRawRevert)
for x in range(questions//5):
    stackAns = finalAns[x*5:x*5+5]
    stackIndex = myIndex[x*5:x*5+5]
    stackGrading = grading[x*5:x*5+5]
    rawBoxes[x] = sp.showAnswer(rawBoxes[x], stackIndex, stackAns, stackGrading, 5, 4)
x = 2
#cv2.imshow('boxpiece', rawBoxes[x*4])
#cv2.imshow('boxpiece1', rawBoxes[x*4+1])
#cv2.imshow('boxpiece2', rawBoxes[x*4+2])
#cv2.imshow('boxpiece3', rawBoxes[x*4+3])
reRawImg = sp.restoreImg(imgRawRevert, rawBoxes, stack)
#cv2.namedWindow("inv", cv2.WINDOW_NORMAL)
#cv2.imshow("resore",reRawImg)
imgRawRevert1 = np.zeros_like(imgWarpColored)
imgRawRevert1[per_y1:per_y2, per_x1:per_x2] = reRawImg
#cv2.imshow("revert",imgRawRevert1)
invMatrix = cv2.getPerspectiveTransform(pt2, pt1)

imgInvWarp = cv2.warpPerspective(
    imgRawRevert1,
    invMatrix,
    (widthImg, heightImg),
    flags=cv2.INTER_NEAREST
)
kernel = np.ones((3,3), np.uint8)
imgInvWarp = cv2.dilate(imgInvWarp,kernel,iterations=1)
gray = cv2.cvtColor(imgInvWarp, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
maskInv = cv2.bitwise_not(mask)
imgBg = cv2.bitwise_and(imgFinal,imgFinal,mask=maskInv)
imgFg = cv2.bitwise_and(imgInvWarp,imgInvWarp,mask=mask)
imgFinal = cv2.add(imgBg, imgFg)
#cv2.imshow('inv', imgInvWarp)

imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
check = cv2.imwrite(r"C:\review_OMR\images\final2020x1.jpg", imgFinal)
print(check)
if cv2.waitKey(0) == ord('x'):
    cv2.destroyAllWindows()
