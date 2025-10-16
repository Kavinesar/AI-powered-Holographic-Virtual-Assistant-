import cv2
import requests
import base64
from PIL import Image
from io import BytesIO
import time

API_URL = "" #endpoint
headers = {"Authorization": "Bearer hf_"} #hf_url

def query(image, question):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "inputs": {
            "image": image_base64,
            "question": question
        }
    }
    while True:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            time.sleep(5)
        else:
            print("Error:", response.json())
            break

def generate_dynamic_response(question, answer):
    return f"If your question is '{question}', then my answer is: '{answer}'."

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Laura Launched")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    cv2.imshow('Webcam Feed', frame)
    key = cv2.waitKey(100) & 0xFF

    if key == ord('c'):
        print("Captured Image. Please type your question in the terminal:")
        question = input("Your question: ")
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        result = query(pil_image, question)
        if result and isinstance(result, list) and 'answer' in result[0]:
            answer = result[0]['answer']
            dynamic_response = generate_dynamic_response(question, answer)
            print("Response:", dynamic_response)
    
    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
