import csv
import cv2
import os


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def takeImages():
    Id = input("Enter Your Id: ")
    name = input("Enter Your Name: ")

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = "FRAS/haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                # incrementing sample number
                sampleNum += 1
                # saving the captured face in the dataset folder TrainingImage
                folder_path = "TrainingImage"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                cv2.imwrite(
                    os.path.join(
                        folder_path, f"{name}.{Id}.{sampleNum}.jpg"
                    ),
                    gray[y:y + h, x:x + w],
                )
                # display the frame
                cv2.imshow("frame", img)
            # wait for 100 milliseconds
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            # break if the sample number is more than 100
            elif sampleNum > 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        print(f"Images Saved for ID: {Id}, Name: {name}")

        # Create or append to the CSV file
        csv_path = "StudentDetails/StudentDetails.csv"
        folder_path = os.path.dirname(csv_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        header = ["Id", "Name"]
        row = [Id, name]

        if not os.path.exists(csv_path):
            # Create a new file and write the header
            with open(csv_path, "w", newline="") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(header)
                writer.writerow(row)
        else:
            # Append to the existing file
            with open(csv_path, "a", newline="") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
    else:
        if not is_number(Id):
            print("Error: Enter a numeric ID.")
        if not name.isalpha():
            print("Error: Enter a valid alphabetical name.")
