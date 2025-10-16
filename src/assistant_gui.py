import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
import pyttsx3
import threading
import queue

class AssistantGUI:
    def __init__(self, gif_path):
        self.root = tk.Tk()
        self.root.title("CHAD Assistant")
        self.root.geometry("500x500")
        
        # GUI Elements
        self.avatar_label = tk.Label(self.root)
        self.avatar_label.pack(pady=20)
        self.response_label = tk.Label(self.root, text="", wraplength=400, justify="left")
        self.response_label.pack()
        
        # Animation
        self.frames = self._load_gif(gif_path)
        self.current_frame = 0
        self.animation_speed = 50  # ms
        self.is_speaking = False
        self.stop_speaking = False
        
        # Speech Engine
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speak_worker, daemon=True)
        self.speech_thread.start()
        
        # Start animation
        self.show_frame(0)

    def _load_gif(self, path):
        try:
            gif = Image.open(path)
            return [ImageTk.PhotoImage(frame.convert('RGBA')) 
                   for frame in ImageSequence.Iterator(gif)]
        except:
            blank = Image.new('RGBA', (500, 500), (255, 255, 255, 0))
            return [ImageTk.PhotoImage(blank)]

    def show_frame(self, frame_idx):
        if len(self.frames) > 0:
            self.current_frame = frame_idx % len(self.frames)
            self.avatar_label.config(image=self.frames[self.current_frame])
            self.avatar_label.image = self.frames[self.current_frame]

    def update_animation(self):
        if self.is_speaking and not self.stop_speaking:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.show_frame(self.current_frame)
            self.root.after(self.animation_speed, self.update_animation)
        else:
            self.show_frame(0)  # Return to first frame when not speaking

    def _speak_worker(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        while True:
            text = self.speech_queue.get()
            self.is_speaking = True
            self.stop_speaking = False
            self.root.after(0, self.update_animation)
            
            # Process each sentence separately
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            for sentence in sentences:
                if self.stop_speaking:
                    break
                engine.say(sentence)
                engine.runAndWait()
            
            self.is_speaking = False
            self.root.after(0, lambda: self.show_frame(0))

    def speak_response(self, text):
        self.response_label.config(text=text)
        self.speech_queue.put(text)

    def run(self):
        self.root.mainloop()