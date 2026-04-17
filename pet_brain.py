import pyautogui
import os
import pyttsx3
from google import genai
from dotenv import load_dotenv

# Optional imports for Focus mode
try:
    import cv2
    import mediapipe as mp
    FOCUS_MODE_AVAILABLE = True
except ImportError:
    FOCUS_MODE_AVAILABLE = False

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env file")

class PetBrain:
    def __init__(self, mode="default"):
        self.mode = mode
        self.client = genai.Client(api_key=GEMINI_KEY)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Slower for accessibility
        
        # Initialize focus detection if in Focus mode
        self.face_mesh = None
        self.cap = None
        self.last_seen = None
        
        if mode == "focus":
            if not FOCUS_MODE_AVAILABLE:
                raise ImportError("Focus mode requires opencv-python and mediapipe. Install with: pip install -r requirements.txt")
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
            self.cap = cv2.VideoCapture(0)
            self.last_seen = __import__('time').time()

    def is_focusing(self):
        """Check if user is focused (Focus mode only)"""
        if self.mode != "focus" or not self.face_mesh:
            return True  # Always return True in Default mode
        
        import time
        ret, frame = self.cap.read()
        if not ret: return False
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            self.last_seen = time.time()
            return True
        return (time.time() - self.last_seen) < 3

    def study_the_screen(self, question="Explain this study material for a student with ADHD."):
        """Analyze screen for study assistance"""
        pyautogui.screenshot("study_session.png")
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[question, "study_session.png"]
            )
            return response.text
        except Exception as e:
            return f"Error connecting to AI: {e}"

    def text_to_speech(self, text):
        """Read text aloud for accessibility (helpful for ESL learners)"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            print(f"TTS Error: {e}")
            return False

    def translate_text(self, text, target_language='es'):
        """Translate text to help ESL learners (default: Spanish)"""
        try:
            result = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Translate this to {target_language} and keep it concise: {text}"
            )
            return result.text
        except Exception as e:
            return f"Translation error: {e}"

    def stop(self):
        """Cleanup TTS engine and webcam if Focus mode"""
        self.tts_engine.stop()
        if self.cap:
            self.cap.release()

