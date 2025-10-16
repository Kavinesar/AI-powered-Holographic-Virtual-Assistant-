from vc5 import VirtualAssistant
from assistant_gui import AssistantGUI
import cv2
import os
from PIL import Image
import threading

class TextAssistant:
    def __init__(self):
        self.assistant = VirtualAssistant()
        self.gui = AssistantGUI(r"F:\\localai\\apps\\main\\avatar22.gif")
        
        print("CHAD: Hello! I'm CHAD, your personal assistant.")
        self.gui.speak_response("Hello! I'm CHAD, your personal assistant.")

    def _capture_image(self):
        """Handles camera capture with text countdown"""
        print("\nOpening camera. Capturing image in 3 seconds...")
        self.gui.speak_response("Opening camera. Capturing image in 3 seconds...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            self.gui.speak_response("Error - Could not open webcam.")
            return None

        # Show preview window with countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                cv2.imshow("CHAD - Image Preview", frame)
                cv2.waitKey(1000)
                print(f"{i}...")
                self.gui.speak_response(str(i))
        
        # Final capture
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()

        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    def _process_capture_flow(self):
        """Complete text-based image analysis flow"""
        image = self._capture_image()
        if not image:
            return "Failed to capture image", False
        
        print("\nPlease type your question about the image:")
        question = input("> ")
        
        # Use existing vc5 analysis
        result = self.assistant.query_vision(image, question)
        if result and isinstance(result, list) and 'answer' in result[0]:
            return self.assistant.generate_full_sentence(question, result[0]['answer']), False
        return "Could not analyze the image", False

    def run(self):
        while True:
            try:
                print("\nType your command (or 'exit' to quit):")
                command = input("> ").lower()
                
                if "exit" in command or "goodbye" in command:
                    self.gui.speak_response("Goodbye! Have a great day.")
                    self.shutdown()
                    break
                    
                if "capture" in command:
                    response, _ = self._process_capture_flow()
                    self.gui.speak_response(response)
                    continue
                
                # Normal commands
                response, should_exit = self.assistant.process_command(command)
                self.gui.speak_response(response)
                
                if should_exit:
                    self.shutdown()
                    break
                    
            except Exception as e:
                print(f"Error: {e}")
                self.gui.speak_response("Let me try that again")

    def shutdown(self):
        self.gui.root.after(100, self.gui.root.destroy)
        os._exit(0)

if __name__ == "__main__":
    assistant = TextAssistant()
    try:
        # Start text processing in separate thread
        text_thread = threading.Thread(target=assistant.run, daemon=True)
        text_thread.start()
        
        # Run GUI in main thread
        assistant.gui.run()
    except KeyboardInterrupt:
        assistant.shutdown()
    except Exception as e:
        print(f"CHAD: Error - {e}")
        assistant.shutdown()