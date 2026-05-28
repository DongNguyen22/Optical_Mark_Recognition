import cv2
import support2025 as sp
import solveImg as si
def getMDD_MD(img):
    img = cv2.resize(img, (1000,1400))
    imgAns = img.copy()
    #Gray
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Blur
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    #threshold
    imgThresh = cv2.threshold(imgBlur,135,255,cv2.THRESH_BINARY_INV)[1]
    # Canny
    imgCanny = cv2.Canny(imgBlur, 50, 150)
    #cv2.imshow("thresh",imgThresh)
    #contour
    contours,h = cv2.findContours(imgCanny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #loc ra contours tu giac area > 50
    contours = sp.rectContour(contours)
    #cv2.drawContours(img, contours, -1, (0,255,0), 2)
    imgPart_MDD_original = img.copy()
    imgPart_MD_original = img.copy()
    imgPart_MDD_origina,pointContours = si.takeImageMDD(imgPart_MDD_original,contours,[610,620,805,810,890,910],[120,160,410,450])
    per_x1,per_y1 = int(pointContours[0][0]) + 22,int(pointContours[0][1]) + 20
    per_x2 = int(pointContours[1][0])
    per_x4,per_y3 = int(pointContours[3][0]),int(pointContours[2][1]) - 10
    imgPart_1 = imgThresh[per_y1:per_y3,per_x1:per_x2 - 12]
    imgPart_2 = imgThresh[per_y1+2:per_y3,per_x2 + 20:per_x4]
    imgPart_1_clone = img[per_y1:per_y3,per_x1:per_x2 - 12]
    imgPart_2_clone = img[per_y1+2:per_y3,per_x2 + 20:per_x4]
    ans_1 = sp.splitAnsPart3([imgPart_1],10,8)
    ans_2 = sp.splitAnsPart3([imgPart_2],10,4)
    pixel_1 = sp.countPixelPartCol(ans_1[0],10,8)
    pixel_2 = sp.countPixelPartCol(ans_2[0],10,4)
    myIndex_1 = sp.countIndex(pixel_1,8)
    myIndex_2 = sp.countIndex(pixel_2,4)
    part_1_colored = sp.showMDD_MD(imgPart_1_clone,myIndex_1,8,10)
    part_2_colored = sp.showMDD_MD(imgPart_2_clone,myIndex_2,4,10)
    img[per_y1:per_y3,per_x1:per_x2 - 12] = part_1_colored
    img[per_y1+2:per_y3,per_x2 + 20:per_x4] = part_2_colored

    studentID = sp.convertMDD_MD(myIndex_1)
    examID = sp.convertMDD_MD(myIndex_2)
    cv2.putText(img,f"{studentID}",(int(per_x1), int(pointContours[0][1])),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img,f"{examID}",(int(per_x2 + 20), int(pointContours[0][1])),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    return img
