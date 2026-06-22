import cv2
import numpy as np
import supportVACT as sp
import solveImg as si
def Omr_vact(img,ans):
    widthImg = 1000
    heightImg = 1400
    questions = len(ans)
    choices = 4
    stack = (questions + 11) // 12
    finalAns = ans
    img = cv2.resize(img,(widthImg,heightImg))
    imgAns = img.copy()
    imgFinal = img.copy()
    #cv2.namedWindow("inv", cv2.WINDOW_NORMAL)
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
    img,pointContour = si.takeImageAnswer(img,contours,[60,120,800,850],[550,600,1150,1200])
    pointContour[1,0] +=154
    pointContour[2,1] +=6
    pointContour[3,1] +=6
    pointContour[3,0] +=154
    pt1 = np.float32([pointContour[0], pointContour[1], pointContour[2], pointContour[3]])
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(imgAns, matrix, (widthImg, heightImg))
    imgRevert = imgWarpColored.copy()
    #cv2.imshow("Canny",img)
    imgCVT = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgCVT, 135, 255, cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("Canny",imgWarpColored)
    boxes = sp.splitImg(imgThresh)
    #x = 0
    #cv2.imshow('boxpiece', boxes[x*4])
    #cv2.imshow('boxpiece1', boxes[x*4+1])
    #cv2.imshow('boxpiece2', boxes[x*4+2])
    #cv2.imshow('boxpiece3', boxes[x*4+3])
    #answer
    ans = sp.splitAns(boxes,12,4)
    #x = 48
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
        validIndex = np.where(arr > 350)[0]
        if len(validIndex) == 0:
            myIndex.append(-1)
        elif len(validIndex) > 1:
            myIndex.append(-2)
        else:
            myIndex.append(validIndex[0])
    #print(myIndex)
    #grading
    grading = []
    for x in range(questions):
        if( myIndex[x] == finalAns[x] ):
            grading.append(1)
        else:
            grading.append(0)
    #print(grading)
    percent = 0
    if len(finalAns) == 40:
        percent = 0.25
    elif len(finalAns) == 50:
        percent = 0.2
    elif len(finalAns) == 120:
        percent = 10
    score = percent*(sum(grading))
    imgRawRevert = np.zeros_like(imgWarpColored)
    raw_boxes = sp.splitImg(imgRawRevert)
    for x in range((questions + 11) // 12):
        start = x * 12
        end = min(start + 12, questions)

        raw_boxes[x] = sp.showAnswer(
            raw_boxes[x],
            myIndex[start:end],
            finalAns[start:end],
            grading[start:end],
            end - start,
            4
        )
    #x = 0
    #cv2.imshow('boxpiece', raw_boxes[x*4])
    #cv2.imshow('bokpiece1', raw_boxes[x*4+1])
    #cv2.imshow('boxpiece2', raw_boxes[x*4+2])
    #cv2.imshow('boxpiece3', raw_boxes[x*4+3])
    reRawImg = sp.restoreImg(imgRawRevert, raw_boxes, stack)
    #cv2.namedWindow("inv", cv2.WINDOW_NORMAL)
    #cv2.imshow("resore",reRawImg)
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)

    imgInvWarp = cv2.warpPerspective(
    reRawImg,
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
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    return imgFinal,score, myIndex