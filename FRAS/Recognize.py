import datetime
import os
import time
import cv2
import pandas as pd


def recognize_attendance():
    # Đường dẫn tệp
    trainer_path = r"FRAS/TrainingImageLabel/Trainner.yml"
    cascade_path = r"FRAS/haarcascade_frontalface_default.xml"
    student_details_path = r"StudentDetails/StudentDetails.csv"
    attendance_dir = r"FRAS/Attendance"

    # Kiểm tra tệp cần thiết
    if not os.path.exists(trainer_path):
        print(f"File '{trainer_path}' không tồn tại. Vui lòng kiểm tra lại.")
        return
    if not os.path.exists(cascade_path):
        print(f"File '{cascade_path}' không tồn tại. Vui lòng kiểm tra lại.")
        return
    if not os.path.exists(student_details_path):
        print(f"File '{student_details_path}' không tồn tại. Vui lòng kiểm tra lại.")
        return

    # Tạo thư mục Attendance nếu chưa tồn tại
    os.makedirs(attendance_dir, exist_ok=True)

    # Khởi tạo recognizer và cascade
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Đọc thông tin sinh viên
    try:
        df = pd.read_csv(student_details_path)
    except Exception as e:
        print(f"Lỗi khi đọc tệp CSV: {e}")
        return

    # Cấu hình video
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra.")
        return

    cam.set(3, 640)  # Độ rộng video
    cam.set(4, 480)  # Độ cao video
    min_w = 0.1 * cam.get(3)
    min_h = 0.1 * cam.get(4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    try:
        while True:
            success, frame = cam.read()
            if not success:
                print("Không thể đọc từ camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(min_w), int(min_h)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 159, 255), 2)
                id_, conf = recognizer.predict(gray[y:y + h, x:x + w])

                if conf < 100:
                    student_name = df.loc[df['Id'] == id_, 'Name'].values
                    student_name = student_name[0] if len(student_name) > 0 else "Unknown"
                    confidence_text = f"  {100 - conf:.2f}%"
                    display_text = f"{id_} - {student_name}"
                else:
                    id_ = "Unknown"
                    display_text = str(id_)
                    confidence_text = f"  {100 - conf:.2f}%"

                if (100 - conf) > 67:
                    timestamp = time.time()
                    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                    time_stamp = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                    attendance.loc[len(attendance)] = [id_, student_name, date, time_stamp]

                display_text += " [Pass]" if (100 - conf) > 67 else ""
                cv2.putText(frame, display_text, (x + 5, y - 5), font, 1, (255, 255, 255), 2)

                confidence_color = (0, 255, 0) if (100 - conf) > 67 else (0, 0, 255)
                cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 1, confidence_color, 1)

            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
            cv2.imshow('Attendance', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Lỗi trong quá trình nhận diện: {e}")
    finally:
        # Ghi dữ liệu điểm danh
        timestamp = time.time()
        date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        time_stamp = datetime.datetime.fromtimestamp(timestamp).strftime('%H-%M-%S')
        attendance_file = os.path.join(attendance_dir, f"Attendance_{date}_{time_stamp}.csv")
        attendance.to_csv(attendance_file, index=False)
        print(f"Danh sách tham dự đã được lưu vào: {attendance_file}")

        # Giải phóng tài nguyên
        cam.release()
        cv2.destroyAllWindows()