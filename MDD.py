import cv2
import numpy as np
import support2020 as sp
import solveImg as si
def takeMDD_MD(img,cols1,cols2):
    MDD = []
    MD = []
    widthImg,heightImg = 500,700
    img = cv2.resize(img,(widthImg,heightImg))
    imgAns = img.copy()
    #cv2.imshow("original",img)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Blur
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    # Canny
    imgCanny = cv2.Canny(imgBlur, 50, 150)
    #contours
    contours,h = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sp.rectContour(contours)
    pointContour = np.zeros((4,2))
    img,pointContour = si.takeImageAnswer(img,contours,[300,350,460,500],[10,50,150,210])
    #cắt ảnh ra để xử lý vùng đáp án
    perY1,perY2,perX1,perX2 = 60,195,340,442
    imgWarp = imgAns[perY1:perY2,perX1:perX2]
    cv2.imshow("warp1",imgWarp)
    #vùng ảnh mã học sinh
    mdd = imgWarp[0:195,0:60]
    cv2.imshow("mdd",mdd)
    #vùng ảnh mã đề
    md = imgWarp[0:195,68:102]
    cv2.imshow("md",md)
    mddCopy = mdd.copy()
    mdCopy = md.copy()
    #chuyển xám và threshold
    imgCvt1 = cv2.cvtColor(mdd,cv2.COLOR_BGR2GRAY)
    imgCvt2 = cv2.cvtColor(md,cv2.COLOR_BGR2GRAY)
    imgThresh1 = cv2.threshold(imgCvt1,150,255,cv2.THRESH_BINARY_INV)[1]
    imgThresh2 = cv2.threshold(imgCvt2,150,255,cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("thresh1",imgThresh1)
    cv2.imshow("thresh2",imgThresh2)
    #chia làm mảng các đáp án
    ans1 = sp.splitMDD(imgThresh1,10,cols1)
    ans2 = sp.splitMDD(imgThresh2,10,cols2)
    cv2.imshow("ans1",ans1[0])
    cv2.imshow("ans12",ans1[2])

    #mảng pixel
    pixel1 = sp.countPixel(ans1,10,cols1)
    pixel2 = sp.countPixel(ans2,10,cols2)
    print(pixel1)
    #numpy mảng cols cột - rows dòng
    MDD = sp.takeMDD(pixel1,cols1)
    studentID = ''.join(map(str,MDD))
    MD = sp.takeMDD(pixel2,cols2)
    examID = ''.join(map(str,MD))

    # vùng đáp án đã được tô
    mddColor = sp.showMDD(mddCopy,MDD,cols1,10)
    mdColor = sp.showMDD(mdCopy,MD,cols2,10)
    #warp ngược lại ảnh gốc
    imgWarp[0:190,0:60] = mddColor
    imgWarp[0:190,68:102] = mdColor

    imgAns[perY1:perY2,perX1:perX2] = imgWarp
    #vẽ số lên ảnh
    x1 = pointContour[0][0] + 8
    x2 = x1+67
    y1 = pointContour[0][1] + 30
    y2 = y1+10

    cv2.putText(imgAns,f"{studentID}",(int(x1), int(y2)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(imgAns,f"{examID}",(int(x2), int(y2)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2,cv2.LINE_AA)
    return imgAns
    