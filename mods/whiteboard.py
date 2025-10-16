import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
draw = False
prev_x, prev_y = 0, 0
canvas = None

camera_width = 640
camera_height = 480


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

cv2.namedWindow('Virtual Whiteboard', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Virtual Whiteboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

screen_width = cv2.getWindowImageRect('Virtual Whiteboard')[2]
screen_height = cv2.getWindowImageRect('Virtual Whiteboard')[3]

x_scale = screen_width / camera_width
y_scale = screen_height / camera_height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  

    
    frame = cv2.resize(frame, (camera_width, camera_height))

    if canvas is None:
        canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            
            thumb_x, thumb_y = int(thumb_tip.x * camera_width), int(thumb_tip.y * camera_height)
            index_x, index_y = int(index_tip.x * camera_width), int(index_tip.y * camera_height)

            
            scaled_index_x = int(index_x * x_scale)
            scaled_index_y = int(index_y * y_scale)

            
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

            
            if distance < 40:
                draw = True
                if prev_x == 0 and prev_y == 0:  
                    prev_x, prev_y = scaled_index_x, scaled_index_y

                
                cv2.line(canvas, (prev_x, prev_y), (scaled_index_x, scaled_index_y), (0, 0, 0), 5)
                prev_x, prev_y = scaled_index_x, scaled_index_y
            else:
                draw = False
                prev_x, prev_y = 0, 0  

            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    frame_fullscreen = cv2.resize(frame, (screen_width, screen_height))
    blended_frame = cv2.addWeighted(canvas, 0.9, frame_fullscreen, 0.1, 0)

    
    cv2.imshow('Virtual Whiteboard', blended_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
