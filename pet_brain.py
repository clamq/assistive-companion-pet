import cv2
import mediapipe as mp
import time
import pyautogui
from google import genai

# Get your key from aistudio.google.com
GEMINI_KEY = "AIzaSyAp-wFo-jyxE6O-ovYi9O18nlJwRAT7sGg"

class PetBrain:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)
        self.client = genai.Client(api_key=GEMINI_KEY)
        self.last_seen = time.time()

    def is_focusing(self):
        ret, frame = self.cap.read()
        if not ret: return False
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            self.last_seen = time.time()
            return True
        return (time.time() - self.last_seen) < 3 

    def study_the_screen(self, question="Explain this study material for a student with ADHD."):
        pyautogui.screenshot("study_session.png")
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[question, "study_session.png"]
            )
            return response.text
        except Exception as e:
            return f"Error connecting to AI: {e}"

    def stop(self):
        self.cap.release()
