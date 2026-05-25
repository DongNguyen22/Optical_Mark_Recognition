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
def reorder(points):

    points = points.reshape((4,2))

    newPoints = np.zeros((4,1,2), dtype=np.int32)

    add = points.sum(1)

    # top-left
    newPoints[0] = points[np.argmin(add)]

    # bottom-right
    newPoints[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)

    # top-right
    newPoints[1] = points[np.argmin(diff)]

    # bottom-left
    newPoints[2] = points[np.argmax(diff)]

    return newPoints
def splitImg(img , rows,cols ):
    offSetX = 10 if rows == 1 else 20
    boxes = []
    h,w = img.shape[:2]
    piece_width = w // cols
    piece_height = h // rows
    for x in range(cols):
        for y in range(rows):
            x_start = x*piece_width+(0 if x == 0 else 8)
            x_end = x_start+piece_width - (0 if x == cols-1 else 10)
            y_start = y*piece_height
            y_end = y_start+piece_height
            piece = img[y_start:y_end,x_start+offSetX:x_end]
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
    offSetX =[25,45,65,85]
    offSetY =[-10,-30,-60,-80,-110]
    cX = (secW*x + secW//2) + offSetX[x]
    cY = (secH*y + secH//2) + offSetY[y]
    point = [cX,cY]
    return point
def showAnswer(box,myIndex,ans,grading,questions,choices):
    h,w = box.shape[:2]
    sec_w = int(w // questions)
    sec_h = int(h // choices)
    radius = min(sec_w, sec_h) // 3
    for x in range(questions):
        myAns = myIndex[x]
        myPoint = getPointAnswer(sec_w,sec_h,myAns,x)
        if grading[x] == 1:
            myColor = (0,255,0)
        else :
            myColor = (0,0,255)
            correctAns = ans[x]
            correctPoint = getPointAnswer(sec_w,sec_h,correctAns,x)
            if myIndex[x] != -1:
                cv2.circle(box,(correctPoint[0],correctPoint[1]),radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
            else :
                cv2.circle(box,(correctPoint[0],correctPoint[1]),radius,(0,125,255),cv2.FILLED,lineType=cv2.LINE_AA)
        if myIndex[x] != -1:
            cv2.circle(box,(myPoint[0],myPoint[1]),radius,myColor,cv2.FILLED,lineType=cv2.LINE_AA)
    return box
def restoreImg(img,boxes,stack):
    h,w = img.shape[:2]
    rows,cols=6,4
    piece_width = w // cols
    piece_height = h // rows
    count = 0
    for x in range(cols):
        for y in range(rows):
            x_start = x* piece_width + (0 if x == 0 else 15)
            x_end = x_start+piece_width - (0 if x == cols -1 else 15)
            y_start = y*piece_height
            y_end = y_start+piece_height

            width_box = x_end -(x_start+20)
            height_box = y_end - y_start
            resized = cv2.resize(boxes[count],(width_box,height_box))
            img[y_start:y_end,x_start+20 :x_end] = resized
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
def splitMDD(img,rows,cols):
    h,w = img.shape[:2]
    piece_height = int(h // rows)
    piece_width = int(w // cols)
    ans =[]
    for x in range(cols):
        for y in range(rows):
            x_start = x*piece_width
            x_end = x_start+piece_width
            y_start = y*piece_height
            y_end = y_start+piece_height
            piece = img[y_start:y_end,x_start:x_end+2]
            ans.append(piece)
    return ans
def takeMDD(pixel,cols):
    ans =[]
    for i in range(cols):
        arr = pixel[i]
        marked = np.where(arr >= 50)[0]
        if len(marked) == 0:
            ans.append(-1)        
        elif len(marked) == 1:
            digit = int(marked[0])
            ans.append(digit)
        else:
            ans.append(-2)
    return ans
def getPointMdd(secW,secH,x,y):
    cX = x*secW + secW//2
    cY = y*secH + secH //2
    point = [cX,cY]
    return point
def showMDD(img,index,cols,rows):
    h,w = img.shape[:2]
    secW = w // cols
    secH = h // rows
    radius = min(secW, secH) // 3
    for i in range(cols):
        myID = index[i]
        if myID == -2 or myID == -1:
            continue
        myPoint = getPointMdd(secW,secH,i,myID)
        cv2.circle(img,myPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
    return img
def countIndex(pixelValues,questions):
    myIndex_1 =[]
    for i in range(questions):
        arr = pixelValues[i]
        marked = np.where(arr >= 50)[0]
        if len(marked) == 0:
            myIndex_1.append(-1)        
        elif len(marked) == 1:
            digit = int(marked[0])
            myIndex_1.append(digit)
        else:
            myIndex_1.append(-2)
    return myIndex_1
def splitAns2025(img,rows,cols):
    ans = []
    h,w = img.shape[:2]
    print(h,w)
    piece_w = (w-5) // cols
    piece_h= (h-14) // rows
    for x in range(cols):
        col = []
        for y in range(rows):
            x_start = x*piece_w + (5 if x == 0 else 0)
            x_end = x_start + piece_w
            print(x_start,x_end)
            y_start = y*piece_h + (14 if y == 0 else 0)
            y_end = y_start+piece_h
            piece = img[y_start:y_end,x_start:x_end]
            col.append(piece)
        ans.append(col)
    return ans



