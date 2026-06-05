import cv2
# import MDD as md
import support2025 as sp
import solveImg as si
def Omr_2025(img,ans):
    final_ans1 = ans[0]
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
    cv2.imshow("thresh",imgThresh)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.imshow("original",img)
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
    imgPart_1_original,pointContour_1 = si.takeImageAnswer(imgPart_1_original,contours,[50,150,930,970],[400,450,700,750])
    imgPart_2_original,pointContour_2 = si.takeImageAnswer(imgPart_2_original,contours,[50,150,930,970],[680,720,860,900])
    imgPart_3_original,pointContour_3 = si.takeImageAnswer(imgPart_3_original,contours,[50,110,930,970],[860,920,1250,1300])
    cv2.namedWindow("part3", cv2.WINDOW_NORMAL)
    cv2.imshow("part1",imgPart_1_original)
    cv2.imshow("part2",imgPart_2_original)
    cv2.imshow("part3",imgPart_3_original)
    #part_1
    #pointContour_1 = sp.reorder(pointContour_1)
    #print(pointContour_1)
    p1_x1 = int(pointContour_1[0][0]) 
    p1_y1 = int(pointContour_1[0][1]) + 51

    p1_x2 = int(pointContour_1[3][0])
    p1_y2 = int(pointContour_1[3][1]) -17
    #part_2

    p2_x1 = int(pointContour_2[0][0] )
    p2_y1 = int(pointContour_2[0][1] ) + 70

    p2_x2 = int(pointContour_2[3][0])
    p2_y2 = int(pointContour_2[3][1]) -10
    #part_3
    #pointContour_3 = sp.reorder(pointContour_3)

    p3_x1 = int(pointContour_3[0][0]) +10
    p3_y1 = int(pointContour_3[0][1]) +85

    p3_x2 = int(pointContour_3[3][0]) +9
    p3_y2 = int(pointContour_3[3][1]) -14
    imgPart_1 = imgThresh[p1_y1:p1_y2, p1_x1:p1_x2]
    imgPart_1_clone = img[p1_y1:p1_y2, p1_x1:p1_x2]
    #cv2.imshow("p1",imgPart_1_clone)
    imgPart_2 = imgThresh[p2_y1:p2_y2, p2_x1:p2_x2]
    imgPart_2_clone = img[p2_y1:p2_y2, p2_x1:p2_x2]
    kkk = sp.splitImg(imgPart_2,1,4)
    kkk_clone = sp.splitImg(imgPart_2_clone,1,4)
    #cv2.imshow("p2",imgPart_2)
    imgPart_3 = imgThresh[p3_y1:p3_y2, p3_x1:p3_x2]
    imgPart_3_clone = img[p3_y1:p3_y2, p3_x1:p3_x2]
    boxes_1 = sp.splitImg(imgPart_1,1,4)
    boxes_1_clone = sp.splitImg(imgPart_1_clone,1,4)
    #cv2.imshow("box1",boxes_1[0])
    #cv2.imshow("box2",boxes_1[1])
    #cv2.imshow("box3",boxes_1[2])
    #cv2.imshow("box4",boxes_1[3])
    boxes_3 = sp.splitImgPart3(imgPart_3,1,6)
    boxes_3_clone = sp.splitImgPart3(imgPart_3_clone,1,6)
    #cv2.imshow("box1",boxes_3_clone[0])
    #cv2.imshow("box2",boxes_3_clone[1])
    #cv2.imshow("box3",boxes_3_clone[2])
    #cv2.imshow("box4",boxes_3_clone[3])
    #cv2.imshow("box5",boxes_3_clone[4])
    #cv2.imshow("box6",boxes_3_clone[5])
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
    #cv2.imshow("box0",boxes_2[0])
    #cv2.imshow("box1",boxes_2[1])
    #cv2.imshow("box2",boxes_2[2])
    #cv2.imshow("box3",boxes_2[3])
    #cv2.imshow("box4",boxes_2[4])
    #cv2.imshow("box5",boxes_2[5])
    #cv2.imshow("box6",boxes_2[6])
    #cv2.imshow("box7",boxes_2[7])
    #1D
    ans_1 = sp.splitAns(boxes_1,10,4)
    #2D
    ans_2 = sp.splitAnsPart2(boxes_2,4,2)
    #cv2.imshow("ans1",ans_2[2][0])
    #cv2.imshow("ans2",ans_2[2][1])
    #3D
    ans_3 = sp.splitAnsPart3(boxes_3,12,4)
    #cv2.imshow("ans0",ans_3[1][10])
    #cv2.imshow("ans1",ans_3[1][11])
    #cv2.imshow("ans2",ans_3[1][12])
    #cv2.imshow("ans3",ans_3[1][13])
    #cv2.imshow("ans4",ans_3[1][14])
    #cv2.imshow("ans5",ans_3[1][15])
    #cv2.imshow("ans6",ans_3[1][16])
    #cv2.imshow("ans7",ans_3[1][17])
    #cv2.imshow("ans8",ans_3[1][18])
    #cv2.imshow("ans9",ans_3[1][19])
    #cv2.imshow("ans10",ans_3[1][20])
    #cv2.imshow("ans11",ans_3[1][21])
    #h,w = boxes_3[0].shape[:2]
    #boxes_3[0] = cv2.resize(boxes_3[0],(100,h))
    #ans_part3 = sp.splitAns2025(boxes_3[0],12,4)
    #cv2.imshow("box_3",boxes_3[0])
    #cv2.imshow("p3_0",ans_part3[0][0])
    #cv2.imshow("p3_1",ans_part3[0][1])
    #pixelValues_1 = sp.countPixel(ans_1,40,4)
    #myIndex_1 = sp.countIndex(pixelValues_1,40)

    #answer
    #part1
    pixel_1 = sp.countPixelPartRows(ans_1,40,4)
    myIndex_1 = sp.countIndex(pixel_1,40)
    print(myIndex_1)
    #part2
    pixel_2 = []
    for x in range(8):
        arr = ans_2[x]
        part = sp.countPixelPartRows(arr,4,2)
        pixel_2.append(part)
    myIndex_2 = []
    for x in range(8):
        arr = pixel_2[x]
        temp = sp.countIndex(arr,4)
        myIndex_2.append(temp)
    print(myIndex_2)
    #part3
    temp_part3 = []
    for x in range(6):
        arr = ans_3[x]
        temp = sp.countPixelPartCol(arr,12,4)
        temp_part3.append(temp)
    myIndex_3 = []
    for x in range(6):
        arr = temp_part3[x]
        temp = sp.countIndex(arr,4)
        myIndex_3.append(temp)
    print(myIndex_3)
    #check correct - wrong
    grading_1 = []
    for i in range(40):
        if myIndex_1[i] == final_ans1[i]:
            grading_1.append(1)
        else :
            grading_1.append(0)
    #part2
    grading_2 = []
    for i in range(8):
        a = final_ans2[i]
        b = myIndex_2[i]
        col = []
        for x in range(4):
            if a[x] == b[x]:
                col.append(1)
            else:
                col.append(0)
        grading_2.append(col)
    #part3
    grading_3 = []
    for x in range(6):
        if myIndex_3[x] == final_ans3[x]:
            grading_3.append(1)
        else:
            grading_3.append(0)
    score_2 = 0
    for x in range(len(grading_2)):
        temp = sum(grading_2[x])
        if temp == 1:
            score_2+=0.1
        elif temp == 2:
            score_2+=0.25
        elif temp == 3:
            score_2+=0.5
        else:
            score_2+=1
    score = 0.25*(sum(grading_1)) + 0.5*(sum(grading_3)) + score_2
    #show answer

    #part1
    for x in range(len(boxes_1_clone)):
        stackIndex = myIndex_1[x*10:x*10+10]
        stackAns = final_ans1[x*10:x*10+10]
        stackGrading = grading_1[x*10:x*10+10]
        boxes_1_clone[x] = sp.showAnswer(boxes_1_clone[x],stackIndex,stackAns,stackGrading,10,4)
    #cv2.imshow("box1",boxes_1_clone[0])
    #cv2.imshow("box2",boxes_1_clone[1])
    #cv2.imshow("box3",boxes_1_clone[2])
    #cv2.imshow("box4",boxes_1_clone[3])
    #part2 
    for x in range(len(boxes_2_clone)):
        stackIndex = myIndex_2[x]
        stackAns= final_ans2[x]
        stackGrading = grading_2[x]
        boxes_2_clone[x] = sp.showAnswer(boxes_2_clone[x],stackIndex,stackAns,stackGrading,4,2)
    #part 3
    for x in range(6):
        stackIndex = myIndex_3[x]
        stackAns= final_ans3[x]
        boxes_3_clone[x] = sp.showAnswerPart_3(boxes_3_clone[x],stackIndex,stackAns,4,12)
    imgPart1_Final = sp.restoreImg(imgPart_1_clone,boxes_1_clone,1,4)
    img[p1_y1:p1_y2, p1_x1:p1_x2] = imgPart1_Final
    #cv2.imshow("final1",imgPart1_Final)
    for x in range(4):
        image = kkk_clone[x]
        h,w = image.shape[:2]
        middle = int(w//2)
        image[0:h,0:middle-1] = boxes_2_clone[x*2]
        image[0:h,middle:w] = boxes_2_clone[x*2+1]
    imgPart2_Final = sp.restoreImg(imgPart_2_clone,kkk_clone,1,4)
    img[p2_y1:p2_y2, p2_x1:p2_x2] = imgPart2_Final
    #cv2.imshow("final2",imgPart2_Final)
    imgPart3_Final = sp.restoreImg_Part3(imgPart_3_clone,boxes_3_clone,1,6)
    img[p3_y1:p3_y2, p3_x1:p3_x2] = imgPart3_Final
    #cv2.imshow("final",img)
    check = cv2.imwrite(r"C:\review_OMR\images\final12.jpg", img)
    print(check)
    #cv2.imshow("final3",imgPart3_Final)
    #cv2.imshow("box0",boxes_3_clone[0])
    #cv2.imshow("box1",boxes_3_clone[1])
    #cv2.imshow("box2",boxes_3_clone[2])
    #cv2.imshow("box3",boxes_3_clone[3])
    #cv2.imshow("box4",boxes_3_clone[4])
    #cv2.imshow("box5",boxes_3_clone[5])
    #cv2.imshow("box0",boxes_2_clone[0])
    #cv2.imshow("box1",boxes_2_clone[1])
    #cv2.imshow("box2",boxes_2_clone[2])
    #cv2.imshow("box3",boxes_2_clone[3])
    #cv2.imshow("box4",boxes_2_clone[4])
    #cv2.imshow("box5",boxes_2_clone[5])
    #cv2.imshow("box6",boxes_2_clone[6])
    #cv2.imshow("box7",boxes_2_clone[7])
    print(score_2)
    return img,score, (myIndex_1, myIndex_2, myIndex_3)