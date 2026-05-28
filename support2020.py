import cv2
import numpy as np
#area > 50
def rectContour(contours):
    rectCont =[]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 30:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour,0.04*peri,True)
            if len(approx) == 4:
                rectCont.append(contour)
    rectCont = sorted(rectCont,key= cv2.contourArea,reverse=True)
    return rectCont
def splitImg(img):
    boxes = []
    h,w = img.shape[:2]
    piece_width = w // 4
    piece_height = h // 6
    for x in range(4):
        for y in range(6):
            x_start = x*piece_width
            x_end = x_start+piece_width
            y_start = y*piece_height
            y_end = y_start+piece_height
            piece = img[y_start:y_end,x_start+88:x_end]
            boxes.append(piece)
    return boxes
def splitAns(boxes,rows,cols):
    ans = []
    for i in boxes:
        h,w = i.shape[:2]
        piece_width = w // cols
        piece_height = h // rows
        for y in range(rows):
            for x in range(cols):
                x_start = piece_width*x
                x_end = x_start+piece_width
                y_start = piece_height*y
                y_end = y_start+piece_height
                piece = i[y_start:y_end,x_start:x_end]
                ans.append(piece)
    return ans
def resizeImg(img,width,height):
    img = cv2.resize(img,(width,height))
    return img
def getPointAnswer(secW,secH,x,y):
    cX = (secW*x + secW//2) 
    cY = (secH*y + secH//2) 
    point = [cX,cY]
    return point
def showAnswer(box,myIndex,ans,grading,questions,choices):
    h,w = box.shape[:2]
    sec_w = int(w // choices)
    sec_h = int(h // questions)
    radius = min(sec_w, sec_h) // 3
    for x in range(questions):
        myAns = myIndex[x]
        myPoint = getPointAnswer(sec_w,sec_h,myAns,x)
        if myIndex[x] == -2:
            correctAns = ans[x]
            correctPoint = getPointAnswer(sec_w,sec_h,correctAns,x)
            cv2.circle(box,correctPoint,radius,(255,0,0),cv2.FILLED,lineType=cv2.LINE_AA)
        else:
            if grading[x] == 1:
                myColor = (0,255,0)
            else :
                myColor = (0,0,255)
                correctAns = ans[x]
                correctPoint = getPointAnswer(sec_w,sec_h,correctAns,x)
                if myIndex[x] != -1:
                    cv2.circle(box,correctPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
                else :
                    cv2.circle(box,correctPoint,radius,(0,125,255),cv2.FILLED,lineType=cv2.LINE_AA)
            if myIndex[x] != -1:
                cv2.circle(box,myPoint,radius,myColor,cv2.FILLED,lineType=cv2.LINE_AA)
    return box
def restoreImg(img,boxes,stack):
    h,w = img.shape[:2]
    rows,cols=6,4
    piece_width = w // cols
    piece_height = h // rows
    count = 0
    for x in range(cols):
        for y in range(rows):
            x_start = x*piece_width
            x_end = x_start+piece_width
            y_start = y*piece_height
            y_end = y_start+piece_height
            img[y_start:y_end,x_start+88:x_end] = boxes[count]
            count += 1
            if count == stack:
                break
        if count == stack:
            break
    return img
def countPixel(ans,rows,cols):
    countR,countC = 0,0
    pixel = np.zeros((cols,rows))
    for img in ans:
        totalPixel = cv2.countNonZero(img)
        pixel[countC][countR] = totalPixel
        countR+=1
        if countR == rows:
            countC +=1
            countR =0
    return pixel


