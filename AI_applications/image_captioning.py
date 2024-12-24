import cv2
import requests
import base64
from PIL import Image
from io import BytesIO
import time

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"  
headers = {"Authorization": "Bearer API_Key"}  
def query(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "inputs": image_base64
    }

    while True:
    
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()  
        elif response.status_code == 503:  
            print("Model is currently loading. Waiting...")
            time.sleep(5)  
        else:
            print("Error:", response.json())
            break
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("C to get a Caption")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error")
        break

    cv2.imshow('Caption generator', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        caption_result = query(pil_image)  
        if caption_result and isinstance(caption_result, list):
            caption = caption_result[0]['generated_text']  
            print("Caption:", caption)  
        else:
            print("Error generating caption:", caption_result)  
    elif key == ord('q'):
        print("EXITING")
        break
cap.release()
cv2.destroyAllWindows()
