import cv2
# import MDD as md
import support2025 as sp
import solveImg as si
from concurrent.futures import ThreadPoolExecutor, as_completed
def grade_part_1(img,imgClone,ans):
    size_p1 = len(ans)
    stack_p1 = (size_p1+9) // 10
    boxes_1 = sp.splitImg(img,1,4)
    boxes_1_clone = sp.splitImg(imgClone,1,4)
    ans_1 = sp.splitAns(boxes_1,10,4,stack_p1)
    pixel_1 = sp.countPixelPartRows(ans_1,size_p1,4)
    myIndex_1 = sp.countIndex(pixel_1,size_p1)
    #print(pixel_1)
    grading_1 = []
    for i in range(size_p1):
        if myIndex_1[i] == ans[i]:
            grading_1.append(1)
        else :
            grading_1.append(0)
    for x in range(stack_p1):
        start = x * 10
        end = min(start + 10, size_p1)
        stackIndex = myIndex_1[start:end]
        stackAns = ans[start:end]
        stackGrading = grading_1[start:end]
        boxes_1_clone[x] = sp.showAnswer(boxes_1_clone[x],stackIndex,stackAns,stackGrading,end - start,4)
    #print(grading_1)
    imgPart1_Final = sp.restoreImg(imgClone,boxes_1_clone,1,4)
    return imgPart1_Final,myIndex_1, grading_1
def grade_part_2(img,imgClone,ans):
    size_p2 = len(ans)
    kkk = sp.splitImg(img,1,4)
    kkk_clone = sp.splitImg(imgClone,1,4)
    boxes_2= []
    for image in kkk:
        h,w = image.shape[:2]
        middle = int(w//2)
        piece_1 = image[0:h,0:middle-1]
        piece_2 = image[0:h,middle:w]
        boxes_2.append(piece_1)
        boxes_2.append(piece_2)
        boxes_2_clone= []
    for image in kkk_clone:
            h,w = image.shape[:2]
            middle = int(w//2)
            piece_1 = image[0:h,0:middle-1]
            piece_2 = image[0:h,middle:w]
            boxes_2_clone.append(piece_1)
            boxes_2_clone.append(piece_2)
    ans_2 = sp.splitAnsPart2(boxes_2,4,2)
    pixel_2 = []
    for x in range(size_p2):
        arr = ans_2[x]
        part = sp.countPixelPartRows(arr,4,2)
        pixel_2.append(part)
    myIndex_2 = []
    for x in range(size_p2):
        arr = pixel_2[x]
        temp = sp.countIndex(arr,4)
        myIndex_2.append(temp)
    grading_2 = []
    for i in range(size_p2):
        a = ans[i]
        b = myIndex_2[i]
        col = []
        for x in range(4):
            if a[x] == b[x]:
                col.append(1)
            else:
                col.append(0)
        grading_2.append(col)
    for x in range(size_p2):
        stackIndex = myIndex_2[x]
        stackAns= ans[x]
        stackGrading = grading_2[x]
        boxes_2_clone[x] = sp.showAnswer(boxes_2_clone[x],stackIndex,stackAns,stackGrading,4,2)
        boxes_2_clone[x] = sp.showAnswer(boxes_2_clone[x],stackIndex,stackAns,stackGrading,4,2)
    imgPart2_Final = sp.restoreImg(imgClone,kkk_clone,1,4)
    return imgPart2_Final,myIndex_2,grading_2
def grade_part_3(img,imgClone,ans):
    size_p3 = len(ans)
    boxes_3 = sp.splitImgPart3(img,1,6)
    boxes_3_clone = sp.splitImgPart3(imgClone,1,6)
    ans_3 = sp.splitAnsPart3(boxes_3,12,4)
    pixel_p3 = []
    for x in range(size_p3):
        arr = ans_3[x]
        temp = sp.countPixelPartCol(arr,12,4)
        pixel_p3.append(temp)
    myIndex_3 = []
    for x in range(size_p3):
        arr = pixel_p3[x]
        temp = sp.countIndex(arr,4)
        myIndex_3.append(temp)
    grading_3 = []
    for x in range(size_p3):
        if myIndex_3[x] == ans[x]:
            grading_3.append(1)
        else:
            grading_3.append(0)
    for x in range(size_p3):
        stackIndex = myIndex_3[x]
        stackAns= ans[x]
        boxes_3_clone[x] = sp.showAnswerPart_3(boxes_3_clone[x],stackIndex,stackAns,4,12)
    imgPart3_Final = sp.restoreImg_Part3(imgClone,boxes_3_clone,1,6)
    return imgPart3_Final,myIndex_3,grading_3
def Omr_2025(img,ans):
    final_ans1 = ans[0]
    size_p1 = len(final_ans1)
    # Part 2: 8 câu đúng sai
    # Đúng = 0, Sai = 1
    final_ans2 = ans[1]
    final_ans3 = sp.change_to_ans_part3(ans[2])
    choices = 4
    img = cv2.resize(img, (1000,1400))
    imgAns = img.copy()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Gray
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Blur
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    #threshold
    imgThresh = cv2.threshold(imgBlur,135,255,cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("thresh",imgThresh)
    #cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    #cv2.imshow("original",img)
    # Canny
    imgCanny = cv2.Canny(imgBlur, 50, 150)
    imgPart_1_original = img.copy()
    imgPart_2_original = img.copy()
    imgPart_3_original = img.copy()
    #cv2.imshow("Canny",imgCanny)
    #contour
    contours,h = cv2.findContours(imgCanny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #loc ra contours tu giac area > 50
    contours = sp.rectContour(contours)
    #cv2.drawContours(img, contours, -1, (0,255,0), 2)
    #part1
    imgPart_1_original,pointContour_1 = si.takeImageAnswer(imgPart_1_original,contours,[50,150,930,970],[420,450,740,800])
    imgPart_2_original,pointContour_2 = si.takeImageAnswer(imgPart_2_original,contours,[50,150,930,970],[680,720,900,950])
    imgPart_3_original,pointContour_3 = si.takeImageAnswer(imgPart_3_original,contours,[50,110,930,970],[860,920,1250,1300])
    #cv2.namedWindow("part3", cv2.WINDOW_NORMAL)
    #cv2.imshow("part1",imgPart_1_original)
    #cv2.imshow("part2",imgPart_2_original)
    #cv2.imshow("part3",imgPart_3_original)
    #part_1
    #pointContour_1 = sp.reorder(pointContour_1)
    #print(pointContour_1)
    p1_x1 = int(pointContour_1[0][0]) +14
    p1_y1 = int(pointContour_1[0][1]) + 55

    p1_x2 = int(pointContour_1[3][0]) + 20
    p1_y2 = int(pointContour_1[3][1]) -17
    #part_2
    pointContour_1[0,1] +=51
    pointContour_1[3,1] -=17
    p2_x1 = int(pointContour_2[0][0] )
    p2_y1 = int(pointContour_2[0][1] ) + 80

    p2_x2 = int(pointContour_2[3][0])
    p2_y2 = int(pointContour_2[3][1]) -10
    pointContour_2[0,1] +=80
    pointContour_2[1,1] -=10
    #part_3

    p3_x1 = int(pointContour_3[0][0]) +10
    p3_y1 = int(pointContour_3[0][1]) +85

    p3_x2 = int(pointContour_3[3][0]) +5
    p3_y2 = int(pointContour_3[3][1]) -14
    pointContour_3[0,1] +=10
    pointContour_3[1,1] +=85
    pointContour_3[0,1] +=5
    pointContour_3[1,1] -=14
    imgPart_1 = imgThresh[p1_y1:p1_y2, p1_x1:p1_x2]
    imgPart_1_clone = img[p1_y1:p1_y2, p1_x1:p1_x2]
    #cv2.imshow("p1",imgPart_1_clone)
    imgPart_2 = imgThresh[p2_y1:p2_y2, p2_x1:p2_x2]
    imgPart_2_clone = img[p2_y1:p2_y2, p2_x1:p2_x2]
    #cv2.imshow("p2",imgPart_2_clone)
    imgPart_3 = imgThresh[p3_y1:p3_y2, p3_x1:p3_x2]
    imgPart_3_clone = img[p3_y1:p3_y2, p3_x1:p3_x2]
    #cv2.imshow("a3",imgPart_3_clone)

    grading_1,grading_2,grading_3 = 0,0,0
    myIndex_1,myIndex_2,myIndex_3 =[],[],[]
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_1 = executor.submit(grade_part_1,imgPart_1,imgPart_1_clone,final_ans1)
        future_2 = executor.submit(grade_part_2,imgPart_2,imgPart_2_clone,final_ans2)
        future_3 = executor.submit(grade_part_3,imgPart_3,imgPart_3_clone,final_ans3)
        
        imgPart1_Final,myIndex_1, grading_1 = future_1.result()
        imgPart2_Final,myIndex_2, grading_2 = future_2.result()
        imgPart3_Final,myIndex_3, grading_3 = future_3.result()
        img[p1_y1:p1_y2, p1_x1:p1_x2] = imgPart1_Final
        img[p2_y1:p2_y2, p2_x1:p2_x2] = imgPart2_Final
        img[p3_y1:p3_y2, p3_x1:p3_x2] = imgPart3_Final

    #score
    percent_p3 = 0
    if size_p1 == 12:
        percent_p3 = 0.5
    elif size_p1 == 18:
        percent_p3 = 0.25

    score_2 = 0
    temp_p2 =[sum(x) for x in grading_2]
    for i in temp_p2:
        if i == 1:
            score_2 += 0.1
        elif i == 2:
            score_2 += 0.25
        elif i == 3:
            score_2 += 0.5
        elif i == 4:
            score_2 +=1
    score = 0.25*(sum(grading_1)) + percent_p3*(sum(grading_3)) + score_2

    return img,score, (myIndex_1, myIndex_2, myIndex_3)