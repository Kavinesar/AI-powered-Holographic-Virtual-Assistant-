import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.flip(frame, 1)

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
            
            
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            
            pyautogui.moveTo(x, y)

            
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = ((thumb.x - index_finger.x) ** 2 + (thumb.y - index_finger.y) ** 2) ** 0.5

            if distance < 0.05:  
                pyautogui.click()

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
