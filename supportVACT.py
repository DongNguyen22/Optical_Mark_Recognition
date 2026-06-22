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
    piece_width = w // 5
    piece_height = h// 2
    for x in range(5):
        for y in range(2):
            x_start = x*piece_width
            x_end = x_start+piece_width
            y_start = piece_height*y
            y_end = y_start+piece_height
            piece = img[y_start + (54 if y == 0 else 48):y_end - (14 if y == 0 else 10),x_start+32:x_end-44]
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
    sec_w = int(w // 4)
    sec_h = int(h // 12)
    radius = min(sec_w, sec_h) // 2
    for x in range(questions):
        myAns = myIndex[x]
        myPoint = getPointAnswer(sec_w,sec_h,myAns,x)
        if myIndex[x] == -2:
            correctAns = ans[x]
            correctPoint = getPointAnswer(sec_w,sec_h,correctAns,x)
            cv2.circle(box,correctPoint,radius,(255,0,0),cv2.FILLED,lineType=cv2.LINE_AA)
        else:
            if grading[x] == 1:
                cv2.circle(box,myPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
            else :
                myPoint = getPointAnswer(sec_w,sec_h,myIndex[x],x)
                correctAns = ans[x]
                correctPoint = getPointAnswer(sec_w,sec_h,correctAns,x)
                if myIndex[x] != -1:
                    cv2.circle(box,correctPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
                    cv2.circle(box,myPoint,radius,(0,0,255),cv2.FILLED,lineType=cv2.LINE_AA)
                else :
                    cv2.circle(box,correctPoint,radius,(0,125,255),cv2.FILLED,lineType=cv2.LINE_AA)
    return box
def restoreImg(img,boxes,stack):
    h,w = img.shape[:2]
    piece_width = w // 5
    piece_height = h// 2
    count = 0
    for x in range(5):
        for y in range(2):
            x_start = x*piece_width
            x_end = x_start+piece_width
            y_start = piece_height*y
            y_end = y_start+piece_height
            img[y_start + (54 if y == 0 else 48):y_end - (14 if y == 0 else 10),x_start+32:x_end-44] = boxes[count]
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

def splitAnsPart3(boxes, rows, cols):
    ans = []
    for image in boxes:
        h, w = image.shape[:2]
        piece_w = w // cols
        piece_h = h // rows
        part = []
        for x in range(cols):
            for y in range(rows):
                x_start = x * piece_w
                x_end   = x_start + piece_w
                y_start = y * piece_h
                y_end   = y_start + piece_h
                part.append(image[y_start:y_end, x_start:x_end])
        ans.append(part)
    return ans

def countPixelPartCol(ans, rows, cols):
    pixel = np.zeros((cols, rows))
    countR, countC = 0, 0
    for img in ans:
        pixel[countC][countR] = cv2.countNonZero(img)
        countR += 1
        if countR == rows:
            countC += 1
            countR = 0
    return pixel

def countIndex(pixelValues, questions):
    myIndex = []
    for i in range(questions):
        marked = np.where(pixelValues[i] >= 70)[0]
        if   len(marked) == 0: myIndex.append(-1)
        elif len(marked) == 1: myIndex.append(int(marked[0]))
        else:                  myIndex.append(-2)
    return myIndex

def showMDD_MD(img, index, questions, choices):
    h, w = img.shape[:2]
    sec_w = w // questions
    sec_h = h // choices
    radius = min(sec_w, sec_h) // 3
    for x in range(questions):
        if index[x] >= 0:
            cX = sec_w * x + sec_w // 2
            cY = sec_h * index[x] + sec_h // 2
            cv2.circle(img, [cX, cY], radius, (0, 255, 0), cv2.FILLED, lineType=cv2.LINE_AA)
    return img

def convertMDD_MD(arr):
    return ''.join('x' if x < 0 else str(x) for x in arr)