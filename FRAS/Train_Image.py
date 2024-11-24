import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread



# -------------- image labesl ------------------------

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


# ----------- train images function ---------------
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Khởi tạo LBPHFaceRecognizer
    harcascadePath = r"FRAS/haarcascade_frontalface_default.xml"
    trainingImagePath = r"../TrainingImage"
    trainerPath = r"FRAS/TrainingImageLabel/Trainner.yml"

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(trainerPath), exist_ok=True)

    detector = cv2.CascadeClassifier(harcascadePath)
    if detector.empty():
        print(f"Lỗi: Không thể tải Haar Cascade từ {harcascadePath}")
        return

    faces, Id = getImagesAndLabels(trainingImagePath)
    if len(faces) == 0:
        print("Không có hình ảnh nào để train. Vui lòng thêm ảnh vào thư mục TrainingImage.")
        return

    # Huấn luyện mô hình
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainerPath)
    print(f"Huấn luyện hoàn tất. Tệp đã lưu tại {trainerPath}")
    counter_img(trainingImagePath)


# Optional, adds a counter for images trained (You can remove it)
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(f"{imgcounter} Hình ảnh được huấn luyện", end="\r")
        time.sleep(0.008)
        imgcounter += 1
