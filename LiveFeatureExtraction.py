from random import randint
from skimage.feature import greycomatrix,greycoprops
from skimage.measure import label,regionprops
from sklearn.cross_validation import train_test_split
import os
import numpy as np
import cv2
import pandas as pd
import time

haarfile = "haarcascade_frontalface_alt.xml"
facedetectfile = cv2.CascadeClassifier(haarfile)
final = []
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetectfile.detectMultiScale(image_grey)

    for x, y, w, h in faces:
        sub_img = frame[y - 10:y + h + 10, x - 10:x + w + 10]
        os.chdir("../")
        cv2.imwrite(str(randint(0, 10000)) + ".jpg", sub_img)

        # EXTRACTING THE FEATURES NOW FROM THE CROPPED IMAGE OF ONLY FRONTAL FACE         
        #print(int(faces[40, 40]))

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        #sub_img = cv2.resize(sub_img,256,256)
##########################################################################
        r = sub_img[70, 70, 0]
        g = sub_img[70, 70, 1]
        b = sub_img[70, 70, 2]
        # LUMINANCE FACTOR
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
        print('Luminanace :')
        print(int(round(luminance)))

        ### MEAN RGB VALUE FOR LIVENESS DETECTION
        a = np.asarray(sub_img)
        image__mean = np.mean(sub_img, axis=0)
        print('******* THIS IS MEAN OF IMAGE********')
        print(image__mean)
        print('**************************************')
        time.sleep(2)

        ## SKEWNESS - STANDARD DEVIATION,TOTAL DATA_POINTS,MEAN OF DATA
        y_bar = np.mean(sub_img, axis=0)
        print()
        variance = np.var(a)
        sd = np.std(a)

        gray_image = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        data_points = cv2.countNonZero(gray_image)
        if (data_points > 15000):
            print("Success")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 2)
            sub_img = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            os.chdir("C:\\Users\\kanak\\Downloads\\Original_Extracted")
            cv2.imwrite(str(1) + ".jpg", sub_img)
        else:
            print("Failed")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            sub_img = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            os.chdir("C:\\Users\\kanak\\Downloads\\Fake_Extracted")
            cv2.imwrite(str(0) + ".jpg", sub_img)

        time.sleep(3)
        print('variance', int(variance))
        print()
        print('standard deviation', int(sd))
        print()
        print('Data Points:', data_points)
        index = range(1,440)

        columns = np.array([variance,sd,data_points,luminance])
        final.append(columns)
        df = pd.DataFrame(final)
        filepath = "C:\\Users\\kanak\\Downloads\\kanak.xlsx"
        df.to_excel(filepath)
        print("Writing to excel Done")

        ########################################################################
        cv2.imshow("Faces Found", frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()