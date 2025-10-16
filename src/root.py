import cv2
import requests
import base64
import time
import threading
import numpy as np
from PIL import Image
from io import BytesIO
from groq import Groq
from datetime import datetime
import mediapipe as mp
import io
import matplotlib.pyplot as plt
import pyttsx3
import os
import subprocess
import platform
import urllib.parse

class VirtualAssistant:
    def __init__(self):
        # API Keys
        self.HF_API_KEY = ""  # Your Hugging Face API key
        self.GROQ_API_KEY = ""  # Your Groq API key
        self.NEWS_API_KEY = ""  # Your News API key

        # File Search Configuration
        self.SEARCH_FOLDER = r"C:\Users\Admin\Desktop\Search"
        
        # Initialize Groq client
        self.client = Groq(api_key=self.GROQ_API_KEY)
        
        # Image Generation API
        self.IMG_GEN_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
        self.IMG_GEN_HEADERS = {"Authorization": f"Bearer {self.HF_API_KEY}"}
        
        # Hugging Face Vision API
        self.HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base"
        self.HF_HEADERS = {"Authorization": f"Bearer {self.HF_API_KEY}"}
        
        # BLIP Captioning API
        self.BLIP_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        
        # Whiteboard state
        self.whiteboard_running = False

    def get_datetime(self):
        now = datetime.now()
        return now.strftime("Today is %A, %d %B %Y. The time is %I:%M %p.")

    def get_latest_news(self):
        try:
            url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={self.NEWS_API_KEY}"
            response = requests.get(url)
            news_data = response.json()
            if news_data["status"] == "ok":
                articles = news_data["articles"][:3]  
                return "\n".join([f"- {article['title']}" for article in articles])
        except Exception as e:
            return f"Error fetching news: {e}"
        return "Could not fetch news."

    def chat_with_groq(self, prompt):
        try:
            datetime_info = self.get_datetime()
            news = self.get_latest_news()
            full_prompt = f"{datetime_info}\n\n{news}\n\nUser's question: {prompt}"
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant. Respond within 20 words."},
                    {"role": "user", "content": full_prompt}
                ],
                model="llama3-70b-8192"
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def get_image_caption(self, image):
        """Get caption for an image using BLIP model"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {"inputs": image_base64}
        
        while True:
            response = requests.post(self.BLIP_API_URL, headers={"Authorization": f"Bearer {self.HF_API_KEY}"}, json=payload)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                time.sleep(5)
            else:
                return "Error processing image for caption."

    def convert_dot_to_extension(self, filename):
        """Convert 'hello dot jpg' to 'hello.jpg'"""
        return filename.replace(' dot ', '.')

    def open_file(self, filename):
        try:
            # Convert 'dot' to actual dot
            filename = self.convert_dot_to_extension(filename)
            
            # Search for the file in the specified directory
            file_path = None
            for root, dirs, files in os.walk(self.SEARCH_FOLDER):
                if filename in files:
                    file_path = os.path.join(root, filename)
                    break
            
            if file_path is None:
                return f"File '{filename}' not found in {self.SEARCH_FOLDER}"

            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', file_path])
            else:  # Linux and others
                subprocess.run(['xdg-open', file_path])
            return f"Opened: {filename}"
        except Exception as e:
            return f"Error opening file: {e}"

    def search_in_chrome(self, query):
        try:
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            if platform.system() == 'Windows':
                subprocess.run(['start', 'chrome', search_url], shell=True)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', '-a', 'Google Chrome', search_url])
            else:
                subprocess.run(['google-chrome', search_url])
                
            return f"Searching Chrome for: {query}"
        except Exception as e:
            return f"Error performing search: {e}"

    def query_vision(self, image, question):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        payload = {"inputs": {"image": image_base64, "question": question}}

        while True:
            response = requests.post(self.HF_API_URL, headers=self.HF_HEADERS, json=payload)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                time.sleep(5)
            else:
                return "Error processing image."

    def generate_full_sentence(self, question, answer):
        return f"In response to your question '{question}', I see {answer} in the image."

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error - Could not open webcam."

        time.sleep(3)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imshow("CHAD - Image Preview", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption_result = self.get_image_caption(pil_image)
            if caption_result and isinstance(caption_result, list):
                caption = caption_result[0]['generated_text']
                return f"I see: {caption}"
            else:
                return "Could not generate caption for the image."
        else:
            return "Error capturing image."

    def virtual_whiteboard(self):
        self.whiteboard_running = True
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mp_drawing = mp.solutions.drawing_utils

        prev_x, prev_y = 0, 0
        canvas = None

        cap = cv2.VideoCapture(0)
        screen_width, screen_height = 1280, 720

        while cap.isOpened() and self.whiteboard_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  

            if canvas is None:
                canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    thumb_x, thumb_y = int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)
                    index_x, index_y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)

                    distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

                    if distance < 40:
                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = index_x, index_y
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 0), 5)
                        prev_x, prev_y = index_x, index_y
                    else:
                        prev_x, prev_y = 0, 0

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame_fullscreen = cv2.resize(frame, (screen_width, screen_height))
            blended_frame = cv2.addWeighted(canvas, 0.9, frame_fullscreen, 0.1, 0)

            cv2.imshow('Virtual Whiteboard', blended_frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or not self.whiteboard_running:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.whiteboard_running = False

    def query_image_gen(self, payload):
        try:
            response = requests.post(self.IMG_GEN_API_URL, headers=self.IMG_GEN_HEADERS, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.content
            elif response.status_code == 503:
                return "The image generation service is currently unavailable. Please try again later."
            elif response.status_code == 429:
                time.sleep(10)
                return self.query_image_gen(payload)  # Retry after delay
            else:
                error_msg = f"Received error code {response.status_code} from the image generator."
                if response.status_code == 500:
                    error_msg = "There's an internal server error with the image generator."
                return error_msg
                
        except requests.exceptions.Timeout:
            return "The image generation is taking too long. It might be a network issue. Please try again."
        except requests.exceptions.ConnectionError:
            return "I'm having trouble connecting to the image generator. Please check your network connection."
        except requests.exceptions.RequestException as e:
            return f"There was an unexpected error with the image generation: {e}"

    def display_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            return "Image displayed successfully."
        except Exception as e:
            return f"Failed to display the generated image: {e}"

    def process_command(self, user_input):
        user_input = user_input.lower()
        
        if user_input == 'exit':
            return "Goodbye! Have a great day.", True
            
        elif user_input.startswith('open '):
            filename = user_input[5:].strip()
            return self.open_file(filename), False
            
        elif user_input.startswith('search for ') and ' in chrome' in user_input:
            query = user_input[11:user_input.index(' in chrome')]
            return self.search_in_chrome(query), False
            
        elif user_input.startswith("generate an image "):
            prompt = user_input.replace("generate an image ", "", 46).strip()
            if not prompt:
                return "Please tell me what you'd like me to generate.", False
                
            image_bytes = self.query_image_gen({"inputs": prompt})
            
            if image_bytes and isinstance(image_bytes, bytes):
                result = self.display_image(image_bytes)
                return f"Image generation completed successfully! {result}", False
            else:
                return image_bytes or "I couldn't generate the image. Would you like to try with a different prompt?", False
                
        elif user_input == 'capture':
            return self.capture_image(), False
            
        elif user_input == 'unlock whiteboard':
            if not self.whiteboard_running:
                threading.Thread(target=self.virtual_whiteboard, daemon=True).start()
                return "Whiteboard opened. Use your thumb and index finger to draw.", False
            return "Whiteboard is already open.", False
                
        elif user_input == 'close whiteboard':
            self.whiteboard_running = False
            return "Whiteboard closed.", False
            
        else:
            return self.chat_with_groq(user_input), False