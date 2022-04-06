import cv2
import numpy as np
import time
import os
import lib.HandTrackingModule as htm

folderPath = "Resources"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))

header = overlayList[0]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(maxHands=1, detectionCon=0.7, trackCon=0.6)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0
#############################
drawColor = (255, 0, 255)
brushThickness = 15
eraseThickness = 100
#############################
while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList)

        #tips of index and middle fingers
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # 4. If selection mode - index and middle fingers are up
        if fingers[1] and fingers[2]:
            #selection mode
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

        # 5. If drawing mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            # draw mode
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp==0 and yp==0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraseThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraseThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)              #DODAJE bity kolorow do siebie
    img = cv2.bitwise_or(img, imgCanvas)            #z racji ze rysuje czarny to miejsce zamienia na taki kolor jaki ma imgCanvas

    #setting the header image
    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0.7)
    cv2.imshow("Painter", img)
    cv2.waitKey(1)