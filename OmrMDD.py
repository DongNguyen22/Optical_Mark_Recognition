import cv2
import numpy as np
import support2020 as sp20
import support2025 as sp25
import solveImg as si

def readMDD_2020(img):
    img = cv2.resize(img, (1000, 1400))
    imgGray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur   = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThresh = cv2.threshold(imgBlur, 135, 255, cv2.THRESH_BINARY_INV)[1]
    imgCanny  = cv2.Canny(imgBlur, 50, 150)
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm 2 corner marker đặc trưng 2020 (ô vuông ~25×25px)
    # top-mid ≈ (696, 67)  |  bot-mid ≈ (693, 412)
    top_mid = None
    bot_mid = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (350 <= area <= 700):
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if 660 <= x <= 730 and 38 <= y <= 90 and top_mid is None:
            top_mid = (x, y)
        if 660 <= x <= 730 and 390 <= y <= 430 and bot_mid is None:
            bot_mid = (x, y)

    if top_mid and bot_mid:
        cx, cy_top = top_mid
        _,  cy_bot = bot_mid
        mdd_x1, mdd_x2 = cx + 22,  cx + 128   # 6 cột MDD
        md_x1,  md_x2  = cx + 151, cx + 206   # 3 cột MD
        y1, y2 = cy_top + 88, cy_bot - 10
    else:
        mdd_x1, mdd_x2 = 718, 824   # fallback tọa độ cứng
        md_x1,  md_x2  = 847, 902
        y1, y2 = 155, 402

    # Tách vùng ảnh
    imgMDD       = imgThresh[y1:y2, mdd_x1:mdd_x2]
    imgMDD_clone = img[y1:y2, mdd_x1:mdd_x2].copy()
    imgMD        = imgThresh[y1:y2, md_x1:md_x2]
    imgMD_clone  = img[y1:y2, md_x1:md_x2].copy()

    # Nhận dạng
    ans_MDD   = sp20.splitAnsPart3([imgMDD], 10, 6)
    pixel_MDD = sp20.countPixelPartCol(ans_MDD[0], 10, 6)
    myIndex_MDD = sp20.countIndex(pixel_MDD, 6)

    ans_MD    = sp20.splitAnsPart3([imgMD], 10, 3)
    pixel_MD  = sp20.countPixelPartCol(ans_MD[0], 10, 3)
    myIndex_MD = sp20.countIndex(pixel_MD, 3)

    # Tô màu và ghi kết quả
    img[y1:y2, mdd_x1:mdd_x2] = sp20.showMDD_MD(imgMDD_clone, myIndex_MDD, 6, 10)
    img[y1:y2, md_x1:md_x2]   = sp20.showMDD_MD(imgMD_clone,  myIndex_MD,  3, 10)

    studentID = sp20.convertMDD_MD(myIndex_MDD)
    examID    = sp20.convertMDD_MD(myIndex_MD)

    cv2.putText(img, f"{studentID}", (mdd_x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{examID}", (md_x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return img, studentID, examID

def readMDD_2025(img):
    img = cv2.resize(img, (1000, 1400))
    imgGray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur   = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThresh = cv2.threshold(imgBlur, 135, 255, cv2.THRESH_BINARY_INV)[1]
    imgCanny  = cv2.Canny(imgBlur, 50, 150)
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sp25.rectContour(contours)

    # Tìm 4 điểm anchor qua takeImageMDD
    imgCopy = img.copy()
    imgCopy, pts = si.takeImageMDD(imgCopy, contours,
                                   [610, 620, 805, 810, 890, 910],
                                   [120, 160, 410, 450])

    pts_ok = not np.any(pts == 0) and (int(pts[2][1]) - int(pts[0][1]) > 30)

    if pts_ok:
        x1     = int(pts[0][0]) + 22
        y1     = int(pts[0][1]) + 20
        mdd_x2 = int(pts[1][0]) - 12
        md_x1  = int(pts[1][0]) + 20
        md_x2  = int(pts[3][0])
        y2     = int(pts[2][1]) - 10
    else:
        x1,  mdd_x2 = 665, 807     # fallback tọa độ cứng
        md_x1, md_x2 = 833, 903
        y1, y2 = 190, 420

    # Tách vùng ảnh
    imgMDD       = imgThresh[y1:y2, x1:mdd_x2]
    imgMDD_clone = img[y1:y2, x1:mdd_x2].copy()
    imgMD        = imgThresh[y1:y2, md_x1:md_x2]
    imgMD_clone  = img[y1:y2, md_x1:md_x2].copy()

    # Nhận dạng
    ans_MDD   = sp25.splitAnsPart3([imgMDD], 10, 8)
    pixel_MDD = sp25.countPixelPartCol(ans_MDD[0], 10, 8)
    myIndex_MDD = sp25.countIndex(pixel_MDD, 8)

    ans_MD    = sp25.splitAnsPart3([imgMD], 10, 4)
    pixel_MD  = sp25.countPixelPartCol(ans_MD[0], 10, 4)
    myIndex_MD = sp25.countIndex(pixel_MD, 4)

    # Tô màu và ghi kết quả
    img[y1:y2, x1:mdd_x2]   = sp25.showMDD_MD(imgMDD_clone, myIndex_MDD, 8, 10)
    img[y1:y2, md_x1:md_x2] = sp25.showMDD_MD(imgMD_clone,  myIndex_MD,  4, 10)

    studentID = sp25.convertMDD_MD(myIndex_MDD)
    examID    = sp25.convertMDD_MD(myIndex_MD)

    cv2.putText(img, f"{studentID}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{examID}", (md_x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return img, studentID, examID