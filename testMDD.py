import cv2
import OmrMDD as omr  # Import file xử lý MDD của bạn

def test_model_2020():
    print("--- TESTING MẪU 2020 ---")
    # 1. Đọc file ảnh mẫu 2020.jpg
    img = cv2.imread(r"C:\Code VSCode\OMR\Optical_Mark_Recognition\images\2020.jpg")
    if img is None:
        print("Lỗi: Không tìm thấy file 2020.jpg!")
        return

    # 2. Gọi hàm readMDD_2020 từ OmrMDD.py
    img_res, student_id, exam_id = omr.readMDD_2020(img)

    # 3. In kết quả nhận dạng dạng chữ ra Console
    print(f"Mã định danh (SBD) nhận diện được: {student_id}")
    print(f"Mã đề nhận diện được: {exam_id}")

    # 4. Hiển thị ảnh kết quả lên màn hình để kiểm tra vùng tô màu
    cv2.namedWindow("Test MDD 2020", cv2.WINDOW_NORMAL)
    cv2.imshow("Test MDD 2020", img_res)


def test_model_2025():
    print("\n--- TESTING MẪU 2025 ---")
    # 1. Đọc file ảnh mẫu 2025.jpg
    img = cv2.imread(r"C:\Code VSCode\OMR\Optical_Mark_Recognition\images\2025.jpg")
    if img is None:
        print("Lỗi: Không tìm thấy file 2025.jpg!")
        return

    # 2. Gọi hàm readMDD_2025 từ OmrMDD.py
    img_res, student_id, exam_id = omr.readMDD_2025(img)

    # 3. In kết quả nhận dạng dạng chữ ra Console
    print(f"Mã định danh (SBD) nhận diện được: {student_id}")
    print(f"Mã đề nhận diện được: {exam_id}")

    # 4. Hiển thị ảnh kết quả lên màn hình để kiểm tra vùng tô màu
    cv2.namedWindow("Test MDD 2025", cv2.WINDOW_NORMAL)
    cv2.imshow("Test MDD 2025", img_res)


if __name__ == "__main__":
    # Chạy test cả 2 mẫu
    test_model_2020()
    test_model_2025()

    cv2.waitKey(0)
    cv2.destroyAllWindows()