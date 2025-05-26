from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time

classID = 0 # 0 for Fake and 1 for Real
outputFolderPath = 'Dataset/DataCollect'
blurThreshold = 35
confidence = 0.8
save = True

offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingpoint = 6
debug = True

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)
    listBlur = []
    listInfo = []

    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]
            if score > confidence:
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3)
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if w < 0:
                    w = 0
                if h < 0:
                    h = 0

                imgFace = img[y: y+h, x: x+w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)
                ih, iw, _ = img.shape
                xc, yc = x+w/2, y+h/2
                xcn, ycn = round(xc/iw, floatingpoint), round(yc/ih, floatingpoint)
                wn, hn = round(w/iw, floatingpoint), round(h/ih, floatingpoint)

                # print(xcn, ycn, wn, hn)

                if xcn > 1:
                    xcn = 1
                if ycn > 1:
                    ycn = 1
                if wn > 1:
                    wn = 1
                if hn > 1:
                    hn = 1

                listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn}\n')
                cv2.circle(imgOut, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(imgOut, f'Score:{int(score*100)}% Blur:{blurValue}%', (x, y-20), scale=2, thickness=2)
                cvzone.cornerRect(imgOut, (x, y, w, h))
                if debug:
                    cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                    cvzone.putTextRect(img, f'Score:{int(score * 100)}% Blur:{blurValue}%', (x, y-20), scale=2, thickness=2)
                    cvzone.cornerRect(img, (x, y, w, h))
        if save:
            if all(listBlur) and listBlur != []:
                timesnow = time()
                timesnow = str(timesnow).split('.')
                timesnow = timesnow[0]+timesnow[1]
                # print(timesnow)
                cv2.imwrite(f"{outputFolderPath}/{timesnow}.jpg", img)
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timesnow}.txt", 'a')
                    f.write(info)
                    f.close()

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
