import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 210)
cap.set(cv2.CAP_PROP_FPS, 60)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # results = hands.process(imgRGB)
    fps = 1 / (time.time()-pTime)
    pTime = time.time()
    # time.sleep(0.1)
    #print(results.multi_hand_landmarks)

    # if results.multi_hand_landmarks:
    #     for handLms in results.multi_hand_landmarks:
    #         for id, lm in enumerate(handLms.landmark):
    #             h, w, c = img.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

    #         mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)