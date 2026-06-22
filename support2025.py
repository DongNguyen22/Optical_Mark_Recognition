import cv2
import numpy as np
#area > 50
def rectContour(contours):
    rectCont =[]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 50:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour,0.04*peri,True)
            if len(approx) == 4:
                rectCont.append(contour)
    rectCont = sorted(rectCont,key= cv2.contourArea,reverse=True)
    return rectCont
def splitImg(img , rows,cols ):
    boxes = []
    h,w = img.shape[:2]
    piece_width = w // cols
    piece_height = h // rows
    for x in range(cols):
        for y in range(rows):
            x_start = x*piece_width 
            x_end = x_start+piece_width 
            y_start = y*piece_height
            y_end = y_start+piece_height
            piece = img[y_start:y_end,x_start + 30:x_end]
            boxes.append(piece)
    return boxes
def restoreImg(img,box,rows,cols):
    h,w = img.shape[:2]
    piece_width = w // cols
    piece_height = h // rows
    offSetX = 35 if rows == 1 else 20
    count = 0
    for x in range(cols):
        for y in range(rows):
            x_start = x*piece_width 
            x_end = x_start+piece_width 
            y_start = y*piece_height
            y_end = y_start+piece_height
            img[y_start:y_end,x_start + 30:x_end] = box[count]
            count += 1
    return img
def splitImgPart3(img , rows,cols ):
    boxes = []
    h,w = img.shape[:2]
    piece_width = w // cols
    piece_height = h // rows
    for x in range(cols):
        for y in range(rows):
            x_start = x*piece_width
            x_end = x_start+piece_width -12
            y_start = y*piece_height
            y_end = y_start+piece_height
            piece = img[y_start:y_end,x_start+40:x_end]
            boxes.append(piece)
    return boxes
def restoreImg_Part3(img,box,rows,cols):
    h,w = img.shape[:2]
    piece_width = w // cols
    piece_height = h // rows
    count = 0
    for x in range(cols):
        for y in range(rows):
            x_start = x*piece_width
            x_end = x_start+piece_width -12
            y_start = y*piece_height
            y_end = y_start+piece_height
            img[y_start:y_end,x_start+40:x_end] = box[count]
            count+=1
    return img
def splitAns(boxes,rows,cols,box_count):
    ans = []
    for k in range(box_count):
        i = boxes[k]
        h,w = i.shape[:2]
        piece_width = w // cols
        piece_height = h // rows
        for y in range(rows):
            for x in range(cols):
                x_start = piece_width*x
                x_end = x_start+piece_width
                y_start = piece_height*y
                y_end = y_start+piece_height
                piece = i[y_start:y_end,x_start:x_end - (10 if rows == 10 else 0)]
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
def getPointAnswer_2025(secW,secH,x,y):
    #offSetX =[25,45,65,85]
    #offSetY =[-10,-30,-60,-80,-110]
    cX = (secW*x + secW//2) 
    cY = (secH*y + secH//2) 
    point = [cX,cY]
    return point
def showAnswer(box,myIndex,ans,grading,questions,choices):
    h,w = box.shape[:2]
    sec_w = int(w // choices) - (5 if choices == 4 else 0)
    sec_h = int(h // (10 if choices == 4 else questions))
    radius = min(sec_w, sec_h) // 3
    for x in range(questions):
        myAns = myIndex[x]
        myPoint = getPointAnswer_2025(sec_w,sec_h,myAns,x)
        if myIndex[x] == -2:
            correctAns = ans[x]
            correctPoint = getPointAnswer_2025(sec_w,sec_h,correctAns,x)
            cv2.circle(box,correctPoint,radius,(255,0,0),cv2.FILLED,lineType=cv2.LINE_AA)
        else:
            if grading[x] == 1:
                cv2.circle(box,myPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
            else :
                correctAns = ans[x]
                correctPoint = getPointAnswer_2025(sec_w,sec_h,correctAns,x)
                if myIndex[x] != -1:
                    cv2.circle(box,correctPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
                    cv2.circle(box,myPoint,radius,(0,0,255),cv2.FILLED,lineType=cv2.LINE_AA)
                else :
                    cv2.circle(box,correctPoint,radius,(0,125,255),cv2.FILLED,lineType=cv2.LINE_AA)
    return box
def countPixelPartCol(ans,rows,cols):
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
def countPixelPartRows(ans,rows,cols):
    countR,countC = 0,0
    pixel = np.zeros((rows,cols))
    for img in ans:
        totalPixel = cv2.countNonZero(img)
        pixel[countR][countC] = totalPixel
        countC+=1
        if countC == cols:
            countR +=1
            countC =0
        if countR == rows:
            break
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
        marked = np.where(arr >= 100)[0]
        if len(marked) == 0:
            myIndex_1.append(-1)        
        elif len(marked) == 1:
            digit = int(marked[0])
            myIndex_1.append(digit)
        else:
            myIndex_1.append(-2)
    return myIndex_1
def splitAnsPart2(boxes,rows,cols):
    ans = []
    for image in boxes:
        h,w = image.shape[:2]
        piece_w = w // cols
        piece_h = h // rows
        col = []
        for y in range(rows):
            for x in range(cols):
                x_start = x*piece_w
                x_end = x_start+ piece_w
                y_start = y*piece_h
                y_end = y_start+piece_h
                piece = image[y_start:y_end,x_start:x_end]
                col.append(piece)
        ans.append(col)
    return ans
def splitAnsPart3(boxes,rows,cols):
    ans = []
    for image in boxes:
        h,w = image.shape[:2]
        piece_w = w // cols
        piece_h = h // rows
        part = []
        for x in range(cols):
            col = []
            for y in range(rows):
                x_start = x*piece_w
                x_end = x_start+ piece_w
                y_start = y*piece_h
                y_end = y_start+piece_h
                piece = image[y_start:y_end,x_start:x_end]
                col.append(piece)
            part = part + col
        ans.append(part)
    return ans
def char_to_index(ch):
    if ch == '-':
        return 0
    elif ch == ',' or ch == ".":
        return 1
    else:
        return int(ch) + 2
def change_to_ans_part3(answers):
    all_indexes = []
    for ans in answers:
        indexes = [char_to_index(ch) for ch in ans]
        all_indexes.append(indexes)
    return all_indexes
def showAnswerPart_3(box,index,ans,questions,choices):
    h,w = box.shape[:2]
    sec_w = w // questions
    sec_h = h // choices
    radius = min(sec_w, sec_h) // 3
    max_len = max(len(index), len(ans))
    for x in range(max_len):
        has_index = x < len(index)
        has_ans = x < len(ans)
        # tô dư
        if has_index and not has_ans:
            if index[x] >= 0:
                myPoint = getPointAnswer_2025(sec_w,sec_h,x,index[x] )
                cv2.circle(box,myPoint,radius,(0,0,255),cv2.FILLED,lineType=cv2.LINE_AA)
        # tô thiếu
        elif has_ans and not has_index:
            correctPoint = getPointAnswer_2025(sec_w,sec_h,x,ans[x])
            cv2.circle(box,correctPoint,radius,(0,125,255),cv2.FILLED,lineType=cv2.LINE_AA)
        # có cả 2
        elif has_index and has_ans:
            # tô nhiều
            if index[x] == -2:
                correctPoint = getPointAnswer_2025(sec_w,sec_h,x,ans[x])
                cv2.circle(box,correctPoint,radius,(255,0,0),cv2.FILLED,lineType=cv2.LINE_AA)
            # bỏ trống
            elif index[x] == -1:
                correctPoint = getPointAnswer_2025(sec_w,sec_h,x,ans[x])
                cv2.circle(box,correctPoint,radius,(0,125,255),cv2.FILLED,lineType=cv2.LINE_AA)
            # đúng
            elif index[x] == ans[x]:
                myPoint = getPointAnswer_2025(sec_w,sec_h,x,index[x])
                cv2.circle(box,myPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
            # sai
            else:
                # đáp án học sinh
                myPoint = getPointAnswer_2025(sec_w,sec_h,x,index[x])
                cv2.circle(box,myPoint,radius,(0,0,255),cv2.FILLED,lineType=cv2.LINE_AA)
                # đáp án đúng
                correctPoint = getPointAnswer_2025(sec_w,sec_h,x,ans[x])
                cv2.circle(box,correctPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
    return box
def showMDD_MD(img,index,questions,choices):
    h,w = img.shape[:2]
    sec_w = w // questions
    sec_h = h // choices
    radius = min(sec_w, sec_h) // 3
    for x in range(questions):
        if index[x] >= 0:
            myPoint = getPointAnswer_2025(sec_w,sec_h,x,index[x])
            cv2.circle(img,myPoint,radius,(0,255,0),cv2.FILLED,lineType=cv2.LINE_AA)
    return img
def convertMDD_MD(arr):
    result = ""
    for x in arr:
        if x == -1 or x == -2:
            result += "x"
        else:
            result += str(x)
    return result


                    



