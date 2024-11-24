import cv2
import os

def camer():
    # Load the cascade
    cascade_path = 'FRAS/haarcascade_frontalface_default.xml'

    # Kiểm tra tệp cascade tồn tại
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit the webcam preview.")

    while True:
        # Read the frame
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)

        # Display the output
        cv2.imshow('Webcam Check', img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
