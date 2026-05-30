import cv2
import numpy as np
import support2020 as sp
import solveImg as si
import MDD as mdd
path = r'C:\review_OMR\images\2020.jpg'
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
cv2.imshow("originally",img)
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
img,pointContour = si.takeImageAnswer(img,contours,[10,60,460,500],[150,210,640,700])
#print(pointContour)
#cv2.imshow("draw 4 contour",img)
#lam phang va khoanh vung  dap an
pt1 = np.float32([pointContour[0], pointContour[1], pointContour[2], pointContour[3]])
pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgWarpColored = cv2.warpPerspective(imgAns, matrix, (widthImg, heightImg))
cv2.imshow('blank', imgWarpColored)
perX1,perX2,perY1,perY2 = 20,490,35,685
imgPer = imgWarpColored[perY1:perY2,perX1:perX2]
#cv2.imshow("per",imgPer)
#threshold
imgRevert = imgPer.copy()
imgCvt = cv2.cvtColor(imgPer,cv2.COLOR_BGR2GRAY)
imgThresh = cv2.threshold(imgCvt,135,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow("imgThresh",imgThresh)
#chia ra lam 24 stack
boxes = sp.splitImg(imgThresh,6,4)
#cv2.imshow("boxes",boxes[0])
ans = sp.splitAns(boxes,5,4)
#cv2.imshow("answer",ans[0])
#tao mang dem so pixel cua tung o dap an
pixelValue = np.zeros((120,choices))
countC,countR = 0,0
for image in ans:
    totalPixel = cv2.countNonZero(image)
    pixelValue[countR][countC] = totalPixel
    countC += 1
    if countC == choices:
        countC =0
        countR += 1
myIndex = []
for i in range(questions):
    arr = pixelValue[i]
    myIndexVal = np.where( arr == np.amax(arr))
    if np.amax(arr) > 50:
        myIndex.append(myIndexVal[0][0])
    else :
        myIndex.append(-1)
grading = []
for i in range(questions):
    if myIndex[i] == finalAns[i]:
        grading.append(1)
    else :
        grading.append(0)
#bat dau to dap an
imgRawRevert = np.zeros_like(imgRevert)
rawBoxes = sp.splitImg(imgRawRevert,6,4)
for x in range(questions//5):
    stackAns = finalAns[x*5:x*5+5]
    stackIndex = myIndex[x*5:x*5+5]
    stackGrading = grading[x*5:x*5+5]

    oriWidth = rawBoxes[x].shape[1]
    oriHeight = rawBoxes[x].shape[0]
    rawBoxes[x] = sp.resizeImg(rawBoxes[x], 500, 500)
    rawBoxes[x] = sp.showAnswer(rawBoxes[x], stackIndex, stackGrading, stackAns, 5, 4)
    rawBoxes[x] = sp.resizeImg(rawBoxes[x], oriWidth, oriHeight)
reRawImg = sp.restoreImg(imgRawRevert, rawBoxes, stack)

imgRawRevert1 = np.zeros_like(imgWarpColored)

imgRawRevert1[perY1:perY2, perX1:perX2] = reRawImg

invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
imgInvWarp = cv2.warpPerspective(imgRawRevert1, invMatrix, (widthImg, heightImg))
#cv2.imshow('inv', imgInvWarp)

imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
#cv2.imshow('step1', imgFinal)
imgFinal = mdd.takeMDD_MD(imgFinal,6,3)
score = sum(grading)
cv2.putText(imgFinal,f"{score}/{questions}",(int(100), int(200)),cv2.FONT_HERSHEY_SIMPLEX,2,(120,120,255),2,cv2.LINE_AA)
cv2.imshow("final",imgFinal)
if cv2.waitKey(0) == ord('x'):
    cv2.destroyAllWindows()
