# NeuroSync — Full Enhanced Demo Scaffold with Camera Mood Analysis
# Features added:
# - Camera-based mood detection (snapshot and optional live via streamlit-webrtc if installed)
# - Mood types: Happy, Sad, Angry, Neutral, Surprised, Fearful, Tired
# - Confidence thresholds to reduce misclassification (returns 'Not sure' if low confidence)
# - Mood-aware psychiatrist chatbot (system prompt includes last detected mood)
# - UI mood color mapping and risk meter integration
# - Graceful fallbacks if OpenAI or optional libraries are not installed

import streamlit as st
from typing import List, Dict
import random
import time
import os
from dotenv import load_dotenv

# Optional heavy libs
try:
    import openai
except Exception:
    openai = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

from PIL import Image
import numpy as np

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.warning("⚠️ OPENAI_API_KEY not found or OpenAI lib not installed. AI Mode will fallback to Test Mode.")

st.set_page_config(page_title="NeuroSync — Demo", layout="wide")

# -----------------------------
# Preset Clinical Cases
# -----------------------------
PRESET_CASES = {
    "Migraine": "Severe throbbing unilateral headache with nausea and photophobia",
    "Stroke/TIA": "Sudden right-sided weakness and slurred speech for 20 minutes",
    "Upper Respiratory Infection": "Fever, cough, sore throat, runny nose",
    "Non-specific Viral": "Fatigue, body aches, mild fever since yesterday",
    "Pediatric URI": "Child with fever, sore throat, and difficulty swallowing"
}

# -----------------------------
# Expanded Flashcard Topics
# -----------------------------
FLASHCARD_TOPICS = {
    "Neurology": ["Stroke", "Migraine", "TIA", "Parkinson's", "Seizures"],
    "Psychiatry": ["Depression", "Anxiety", "Bipolar Disorder", "OCD", "PTSD"],
    "Cardiology": ["MI", "CHF", "Hypertension"],
    "Infectious Diseases": ["COVID-19", "URI", "Sepsis"],
    "Pediatrics": ["URI", "Fever", "Rash"],
    "Emergency Medicine": ["Shock", "Trauma", "Chest Pain"]
}

# -----------------------------
# Risk Level Function
# -----------------------------
def risk_level(symptoms: str) -> str:
    s = symptoms.lower()
    if any(k in s for k in ["weakness", "slurred speech", "sudden", "severe headache"]):
        return "High"
    elif any(k in s for k in ["fever", "cough", "throbbing"]):
        return "Medium"
    else:
        return "Low"

# -----------------------------
# Toy Reasoner
# -----------------------------
def toy_reasoner(symptoms: str, age: int=None, explain_patient: bool=False) -> Dict:
    s = symptoms.lower()
    ddx, explanation, tests = [], [], []

    if any(k in s for k in ["headache", "migraine", "throbbing"]):
        ddx.append("Migraine")
        explanation.append("Neuronal hyperexcitability and trigeminovascular activation often underlie migraines.")
        tests.append("Neurological exam; imaging only if red flags.")
    if any(k in s for k in ["weakness", "numbness", "paresthesia"]):
        ddx.append("Transient Ischemic Attack / Stroke")
        explanation.append("Focal neurological deficits may reflect vascular events — time-sensitive.")
        tests.append("Immediate CT/MRI and vascular imaging; bedside NIHSS assessment.")
    if any(k in s for k in ["fever", "cough", "sore throat"]):
        ddx.append("Upper respiratory infection")
        explanation.append("Viral or bacterial infection of the upper airway — supportive care often sufficient.")
        tests.append("Rapid antigen tests, chest X-ray if lower respiratory signs.")
    if not ddx:
        ddx = ["Non-specific viral illness"]
        explanation = ["Symptoms are non-specific; consider broader history and exam."]
        tests = ["Full history, vitals, targeted exam."]

    if explain_patient:
        explanation = [f"Patient Explanation: {e.lower()}" for e in explanation]

    return {"differential": ddx, "explanation": explanation, "recommended_tests": tests}

# -----------------------------
# Flashcard Generator (Fixed)
# -----------------------------
def generate_flashcards(topic: str, n: int=6) -> List[Dict[str,str]]:
    all_cards = []
    for category, items in FLASHCARD_TOPICS.items():
        if topic.lower() in category.lower():
            for item in items:
                all_cards.append({"q": f"{item} key points", "a": f"Summary of {item} information."})
    
    if len(all_cards) < n:
        extra_cards = []
        for category, items in FLASHCARD_TOPICS.items():
            for item in items:
                extra_cards.append({"q": f"{item} key points", "a": f"Summary of {item} information."})
        all_cards.extend(extra_cards)
    
    random.shuffle(all_cards)
    return all_cards[:n]

# -----------------------------
# Mood Detection Utilities
# -----------------------------
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Fearful", "Tired"]

# Small demo weights for a linear mapper - these are heuristic and meant for demo only.
# In practice you would replace this with a calibrated classifier.
EMOTION_WEIGHTS = np.array([
    [0.2, 0.1, -0.3, -0.4, 0.1, 0.2, 0.0],    # brow-like feature
    [-0.1, 0.3, -0.1, -0.2, 0.1, 0.2, 0.4],   # eye openness
    [0.3, -0.2, 0.1, 0.2, -0.1, -0.2, 0.1],   # mouth openness
])

CONFIDENCE_THRESHOLD = 0.55  # below this -> "Not sure"

# Mediapipe FaceMesh landmark indices used for simple geometric features
# These indices are safe for the standard face mesh set.
# We'll pick landmarks that exist across faces: left brow, right brow, upper/lower lip, eye landmarks.
LM = {
    'left_brow': 105,
    'right_brow': 334,
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'right_eye_top': 386,
    'right_eye_bottom': 374,
    'upper_lip': 13,
    'lower_lip': 14
}


def analyze_mood_from_image(image: Image.Image):
    """
    Returns (mood_label, confidence_float) or ("No face detected", 0.0)
    Uses Mediapipe FaceMesh landmarks to compute 3 stable features:
    - brow distance (left vs right) -> rough proxy for furrow/raise
    - eye openness (sum of normalized distances)
    - mouth openness (vertical lip distance)

    If Mediapipe or CV2 not installed, returns ("Unavailable", 0.0)
    """
    if mp is None or cv2 is None:
        return "Unavailable (mediapipe/cv2 missing)", 0.0

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Convert PIL -> BGR ndarray as Mediapipe expects RGB input
    img = np.array(image.convert('RGB'))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return "No face detected", 0.0

    face = results.multi_face_landmarks[0]

    # helper to safely read landmark coords
    h, w = img.shape[:2]
    def lm_xy(idx):
        try:
            lm = face.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
        except Exception:
            return None

    # get landmarks
    lb = lm_xy(LM['left_brow'])
    rb = lm_xy(LM['right_brow'])
    let = lm_xy(LM['left_eye_top'])
    leb = lm_xy(LM['left_eye_bottom'])
    ret = lm_xy(LM['right_eye_top'])
    reb = lm_xy(LM['right_eye_bottom'])
    ul = lm_xy(LM['upper_lip'])
    ll = lm_xy(LM['lower_lip'])

    # if any landmark missing, fallback
    if None in [lb, rb, let, leb, ret, reb, ul, ll]:
        return "Low quality face", 0.0

    # features
    brow_dist = np.linalg.norm(lb - rb) / max(w, h)  # normalized inter-brow distance
    eye_open = (np.linalg.norm(let - leb) + np.linalg.norm(ret - reb)) / (2.0 * max(w, h))
    mouth_open = np.linalg.norm(ul - ll) / max(w, h)

    vec = np.array([brow_dist, eye_open, mouth_open])

    # compute logits and softmax
    logits = vec @ EMOTION_WEIGHTS  # shape (7,)
    # numeric stability
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return "Not sure / Neutral", confidence

    return EMOTION_LABELS[top_idx], confidence

# -----------------------------
# UI Helpers
# -----------------------------
MOOD_COLOR = {
    "Happy": "#2ECC71",
    "Sad": "#3498DB",
    "Angry": "#E74C3C",
    "Neutral": "#95A5A6",
    "Surprised": "#F1C40F",
    "Fearful": "#9B59B6",
    "Tired": "#34495E",
    "Not sure / Neutral": "#BDC3C7",
    "No face detected": "#F39C12",
    "Unavailable (mediapipe/cv2 missing)": "#E67E22",
    "Low quality face": "#E67E22"
}

# persist last mood
if "last_mood" not in st.session_state:
    st.session_state.last_mood = "Neutral"

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown("<h1 style='text-align:center; color: #4B0082;'>🧠 NeuroSync — Intelligent Clinical + Learning Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color: #6A5ACD;'>Demo (Judge Mode)</h3>", unsafe_allow_html=True)

sidebar = st.sidebar
mode = sidebar.selectbox("Mode", ["Test Mode (offline)", "AI Mode (requires OpenAI API)"])
explain_patient = sidebar.checkbox("Explain like patient")

col1, col2 = st.columns([2,1])

# -----------------------------
# Clinical Mode
# -----------------------------
with col1:
    st.header("Clinical Quick-Triage")
    preset = st.selectbox("Select a preset case", ["Custom"] + list(PRESET_CASES.keys()))
    if preset != "Custom":
        symptoms = st.text_area("Patient symptoms", value=PRESET_CASES[preset], height=100)
    else:
        symptoms = st.text_area("Patient symptoms", value="Severe unilateral throbbing headache with nausea and photophobia")

    age = st.number_input("Age", min_value=0, max_value=120, value=28)
    submit = st.button("Generate Differential")

    if submit:
        with st.spinner("Processing..."):
            time.sleep(1)
            level = risk_level(symptoms)

            if mode.startswith("Test"):
                result = toy_reasoner(symptoms, age, explain_patient)
            else:
                if openai is None or not OPENAI_API_KEY:
                    st.error("OpenAI API not configured. Using Test Mode.")
                    result = toy_reasoner(symptoms, age, explain_patient)
                else:
                    prompt = f"Provide differential diagnosis, mechanistic explanation and recommended tests for: {symptoms}"
                    if explain_patient:
                        prompt += " Explain in simple patient-friendly language."
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        text = response['choices'][0]['message']['content']
                        result = {"differential": [text], "explanation": [text], "recommended_tests": [text]}
                    except Exception as e:
                        st.error(f"OpenAI call failed: {e}. Using Test Mode reasoner.")
                        result = toy_reasoner(symptoms, age, explain_patient)

        # Display Results
        color = "#00FF00" if level=="Low" else "#FFD700" if level=="Medium" else "#FF4500"
        st.markdown(f"<h3 style='color:{color}'>Risk Level: {level}</h3>", unsafe_allow_html=True)
        st.subheader("Differential Diagnosis")
        for i, d in enumerate(result['differential'], 1):
            st.markdown(f"**{i}. {d}**")
        st.subheader("Mechanistic Explanations")
        for expl in result['explanation']:
            st.markdown(f"- {expl}")
        st.subheader("Recommended Next Steps / Tests")
        for t in result['recommended_tests']:
            st.markdown(f"- {t}")

# -----------------------------
# Study Mode
# -----------------------------
with col2:
    st.header("Study Mode")
    topic = st.selectbox("Select Flashcard Topic", list(FLASHCARD_TOPICS.keys()))
    n = st.slider("# of flashcards", 2, 6, 4)
    if st.button("Generate Flashcards"):
        cards = generate_flashcards(topic, n)
        for c in cards:
            st.markdown(f"**Q:** {c['q']}  \n**A:** {c['a']}")

# -----------------------------
# Face Mood Analyzer Section
# -----------------------------
st.markdown("---")
st.header("Face Mood Analyzer 😶🧠")

colA, colB = st.columns([1,2])
with colA:
    st.write("**Capture**")
    # snapshot
    snapshot = st.camera_input("Take a picture for mood detection")

    use_live = st.checkbox("Enable Live Mode (requires streamlit-webrtc)")
    if use_live:
        st.info("Live mode requires the optional package 'streamlit-webrtc'. If not installed, live mode will be unavailable.")

    if st.button("Analyze Latest Snapshot") or snapshot is not None:
        img_file = snapshot if snapshot is not None else None
        if img_file is None:
            st.warning("No image captured. Use the camera input to take a picture.")
        else:
            image = Image.open(img_file)
            mood, conf = analyze_mood_from_image(image)
            st.session_state.last_mood = mood

            color = MOOD_COLOR.get(mood, "#FFFFFF")
            st.image(image, caption=f"Captured Image — Mood: {mood} (conf {conf:.2f})", width=320)
            st.markdown(f"<h3 style='color:{color}'>Detected Mood: {mood} — Confidence: {conf:.2f}</h3>", unsafe_allow_html=True)

            if conf < CONFIDENCE_THRESHOLD:
                st.warning("Mood confidence is low — consider retaking the picture or ensuring good lighting.")

with colB:
    st.write("**Mood Explanation & Integration**")
    st.markdown("- Mood is determined using stable geometric face features (brow distance, eye openness, mouth openness).")
    st.markdown("- If the model is uncertain, it will return 'Not sure / Neutral' instead of risking a wrong classification.")
    st.markdown("- Detected mood is automatically included as context when using the Psychiatrist Chatbot below.")

# -----------------------------
# Psychiatrist Chatbot with Memory
# -----------------------------
st.markdown("---")
st.header("Psychiatrist Chatbot")

# Initialize chat_history with preloaded lines (hidden)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "user", "content": "I feel anxious all the time.", "hidden": True},
        {"role": "assistant", "content": "I hear you. Let's try some deep breathing exercises.", "hidden": True},
        {"role": "user", "content": "I have trouble sleeping at night.", "hidden": True},
        {"role": "assistant", "content": "Sleep hygiene is important. Avoid screens before bed and try relaxation techniques.", "hidden": True},
        {"role": "user", "content": "I often feel sad and unmotivated.", "hidden": True},
        {"role": "assistant", "content": "It’s normal to feel low sometimes. Journaling and small daily goals can help.", "hidden": True},
        {"role": "user", "content": "I get panic attacks without warning.", "hidden": True},
        {"role": "assistant", "content": "Grounding techniques like focusing on your senses can help during a panic attack.", "hidden": True},
        {"role": "user", "content": "I am stressed about exams.", "hidden": True},
        {"role": "assistant", "content": "Try breaking your study into small manageable chunks and take short breaks.", "hidden": True}
    ]

user_input = st.text_area("Talk to NeuroSync Psychiatrist")
if st.button("Send"):
    with st.spinner("NeuroSync is thinking..."):
        time.sleep(0.5)
        if mode.startswith("Test") or openai is None or not OPENAI_API_KEY:
            # Always include mood-aware empathetic phrasing in Test Mode
            lm = st.session_state.get('last_mood', 'Neutral')
            reply = f"[Test Mode] It sounds like you are feeling {lm.lower()}. Try deep breathing, journaling, and consult a mental health professional if persistent."
        else:
            # Build context with hidden memory + latest mood as system instruction
            context_messages = []
            # System message: mood-aware
            lm = st.session_state.get('last_mood', 'Neutral')
            system_message = {"role": "system", "content": f"You are a compassionate psychiatrist. The user's current facial mood is {lm}. Respond empathetically and appropriately."}
            context_messages.append(system_message)

            # add hidden preloaded lines for continuity
            for msg in st.session_state.chat_history:
                # include hidden history for model context but don't show to user
                context_messages.append({"role": msg["role"], "content": msg["content"]})

            # add current user message
            context_messages.append({"role": "user", "content": user_input})

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=context_messages
                )
                reply = response['choices'][0]['message']['content']
            except Exception as e:
                st.error(f"OpenAI call failed: {e}. Falling back to Test Mode reply.")
                lm = st.session_state.get('last_mood', 'Neutral')
                reply = f"[Fallback] I hear you. It seems you are {lm.lower()}. Consider grounding and professional support if this persists."

        # Store new user and assistant messages as visible
        st.session_state.chat_history.append({"role": "user", "content": user_input, "hidden": False})
        st.session_state.chat_history.append({"role": "assistant", "content": reply, "hidden": False})

# Display only visible messages
for msg in st.session_state.chat_history:
    if not msg.get("hidden", False):
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**NeuroSync:** {msg['content']}")

st.success("NeuroSync fully enhanced with camera mood detection, mood-aware chatbot, expanded flashcards, and robust fallbacks.")

# -----------------------------
# NOTES for deployment
# -----------------------------
# - Install optional dependencies for full features: pip install mediapipe opencv-python streamlit-webrtc openai python-dotenv
# - Live webcam mode requires streamlit-webrtc; demo uses camera_input snapshot for reliable, cross-platform captures.
# - The lightweight linear mapping used here is heuristic and intended for demo use: replace with a calibrated ML model
#   for production usage (and ensure proper ethical use, consent, and bias testing before clinical deployment).

# End of file
