import cv2
import numpy as np
def takeImageAnswer(img,contours,indexX,indexY):
    ori_x1,ori_x2,ori_x3,ori_x4 = indexX[:4]
    x1,x2,x3,x4 = indexX[:4]
    ori_y1,ori_y2,ori_y3,ori_y4 = indexY[:4]
    y1,y2,y3,y4 = indexY[:4]
    time,a,b,c,d = 0,0,0,0,0
    step = 5
    pointContour = np.zeros((4,2))
    while(a!=1 or b!=1 or c!=1 or d!=1) and time!=10:
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if a!=1:
                x2 = ori_x2+time*step
                y2 = ori_y2+time*step
                if x1<=x<=x2 and x1<=x+w<=x2 and y1<=y<=y2 and y1<=y+h<=y2:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    a =1
                    pointContour[0][0],pointContour[0][1] = x,y
            if c!=1:
                x2 = ori_x2+time*step
                y3 = ori_y3-time*step
                if x1<=x<=x2 and x1<=x+w<=x2 and y3<=y<=y4 and y3<=y+h<=y4:
                    cv2.rectangle(img, (x1, y3), (x2, y4), (0, 255, 0), 2)
                    c =1
                    pointContour[2][0],pointContour[2][1] = x,y
            if b!=1:
                x3 = ori_x3-time*step
                y2 = ori_y2+time*step
                if x3<=x<=x4 and x3<=x+w<=x4 and y1<=y<=y2 and y1<=y+h<=y2:
                    cv2.rectangle(img, (x3, y1), (x4, y2), (0, 255, 0), 2)
                    b =1
                    pointContour[1][0],pointContour[1][1] = x,y
            if d!=1:
                x3 = ori_x3-time*step
                y3 = ori_y3-time*step
                if x3<=x<=x4 and x3<=x+w<=x4 and y3<=y<=y4 and y3<=y+h<=y4:
                    cv2.rectangle(img, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    d =1
                    pointContour[3][0],pointContour[3][1] = x,y
        time+=1
    return img,pointContour
def takeImageAfterWarp(img,contours,indexX,indexY,key):
    ori_x1,ori_x2,ori_x3,ori_x4 = indexX[:4]
    x1,x2,x3,x4 = indexX[:4]
    ori_y1,ori_y2,ori_y3,ori_y4 = indexY[:4]
    y1,y2,y3,y4 = indexY[:4]
    time = 0
    a,b,c,d = key[:4]
    step = 10
    pointContour = np.zeros((4,2))
    while(a!=1 or b!=1 or c!=1 or d!=1) and time!=10:
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if a!=1:
                x2 = ori_x2+time*step
                y2 = ori_y2+time*step
                if x1<=x<=x2 and x1<=x+w<=x2 and y1<=y<=y2 and y1<=y+h<=y2:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    a =1
                    pointContour[0][0],pointContour[0][1] = x,y
            if c!=1:
                x2 = ori_x2+time*step
                y3 = ori_y3-time*step
                if x1<=x<=x2 and x1<=x+w<=x2 and y3<=y<=y4 and y3<=y+h<=y4:
                    cv2.rectangle(img, (x1, y3), (x2, y4), (0, 255, 0), 2)
                    c =1
                    pointContour[2][0],pointContour[2][1] = x,y
            if b!=1:
                x3 = ori_x3-time*step
                y2 = ori_y2+time*step
                if x3<=x<=x4 and x3<=x+w<=x4 and y1<=y<=y2 and y1<=y+h<=y2:
                    cv2.rectangle(img, (x3, y1), (x4, y2), (0, 255, 0), 2)
                    b =1
                    pointContour[1][0],pointContour[1][1] = x,y
            if d!=1:
                x3 = ori_x3-time*step
                y3 = ori_y3-time*step
                if x3<=x<=x4 and x3<=x+w<=x4 and y3<=y<=y4 and y3<=y+h<=y4:
                    cv2.rectangle(img, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    d =1
                    pointContour[3][0],pointContour[3][1] = x,y
        time+=1
    return img,pointContour
def takeImageMDD(img,contours,indexX,indexY):
    ori_x1,ori_x2,ori_x3,ori_x4,ori_x5,ori_x6 = indexX[:6]
    x1,x2,x3,x4,x5,x6 = indexX[:6]
    ori_y1,ori_y2,ori_y3,ori_y4 = indexY[:4]
    y1,y2,y3,y4 = indexY[:4]
    time,a,b,c,d = 0,0,0,0,0
    step = 10
    pointContour = np.zeros((4,2))
    while(a!=1 or b!=1 or c!=1 or d!=1) and time!=10:
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if a!=1:
                x2 = ori_x2+time*step
                y2 = ori_y2+time*step
                if x1<=x<=x2 and x1<=x+w<=x2 and y1<=y<=y2 and y1<=y+h<=y2:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    a =1
                    pointContour[0][0],pointContour[0][1] = x,y
            if c!=1:
                x4 = ori_x4+time*step
                y4 = ori_y4+time*step
                if x3<=x<=x4 and x3<=x+w<=x4 and y3<=y<=y4 and y3<=y+h<=y4:
                    cv2.rectangle(img, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    c =1
                    pointContour[2][0],pointContour[2][1] = x,y
            if b!=1:
                x4 = ori_x4+time*step
                y2 = ori_y2+time*step
                if x3<=x<=x4 and x3<=x+w<=x4 and y1<=y<=y2 and y1<=y+h<=y2:
                    cv2.rectangle(img, (x3, y1), (x4, y2), (0, 255, 0), 2)
                    b =1
                    pointContour[1][0],pointContour[1][1] = x,y
            if d!=1:
                x6 = ori_x6+time*step
                y4 = ori_y4+time*step
                if x5<=x<=x6 and x5<=x+w<=x6 and y3<=y<=y4 and y3<=y+h<=y4:
                    cv2.rectangle(img, (x5, y3), (x6, y4), (0, 255, 0), 2)
                    d =1
                    pointContour[3][0],pointContour[3][1] = x,y
        time+=1
    return img,pointContour