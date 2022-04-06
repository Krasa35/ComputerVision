import cv2
import HandTrackingModule as htm
import time

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands = 4)
while True:
    success, img = cap.read()
    img = detector.findHands(img, )
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        print(lmList[20])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)