import math
import cv2
import time
import numpy as np
import platform

# Check the operating system
is_windows = platform.system() == "Windows"

if is_windows:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
else:
    import alsaaudio

import lib.HandTrackingModule as htm

######################################
wCam, hCam = 640, 480
maxNorm, minNorm = 500, 10
minZ, maxZ = -0.15, -0.02
pTime, percent, norm = 0, 0, 0
hand = htm.handDetector(maxHands=1)

if is_windows:
    devices = AudioUtilities.GetSpeakers()  # VOLUME
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    maxVol, minVol = volume.GetVolumeRange()[1], volume.GetVolumeRange()[0]
else:
    mixer = alsaaudio.Mixer()
    minVol, maxVol = 0, 100  # ALSA volume range is typically 0-100
######################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    success, img = cap.read()

    hand.findHands(img)
    lmList = hand.findPosition(img, draw=False)
    if len(lmList) != 0:
        cv2.line(img, (lmList[4][1], lmList[4][2]), (lmList[8][1], lmList[8][2]), (0, 255, 0), 2)  # DRAW LINE
        norm = math.hypot(lmList[8][1] - lmList[4][1], lmList[8][2] - lmList[4][2])  # FIND DISTANCE
        norm = norm * np.interp(lmList[4][3], [minZ, maxZ], [0.01, 100]) / 10  # DEPENDANCE ON hand.landmark.z

    cTime = time.time()  # FPS
    fps = 1 / (cTime - pTime)
    pTime = cTime

    vol = np.interp(norm, [minNorm, maxNorm], [minVol, maxVol])
    percent = np.interp(norm, [minNorm, maxNorm], [0, 100])

    if is_windows:
        volume.SetMasterVolumeLevel(vol, None)  # VOLUME SET
    else:
        mixer.setvolume(int(vol))  # Set volume for Linux

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img, f'{int(percent)} %', (20, hCam - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.rectangle(img, (20, hCam - 60), (40, hCam - 160), (255, 0, 0), 2)  # PROGRESS BAR
    cv2.rectangle(img, (20, hCam - 60), (40, hCam - int(percent) - 60), (255, 0, 0), cv2.FILLED)

    cv2.imshow("Cam", img)
    if cv2.waitKey(1) >= 0:
        break