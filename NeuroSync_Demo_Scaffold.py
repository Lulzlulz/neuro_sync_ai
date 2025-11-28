"""
NeuroSync — Full Enhanced Demo Scaffold with Real-Time Psychiatrist Chatbot
Features:
- Polished UI with gradient headers, icons, colored risk meter
- Preset clinical cases
- Study Mode with expanded flashcard topics
- Explain Like Patient toggle
- Real AI Mode integration with OpenAI API
- Real-time, context-aware psychiatrist chatbot with memory
"""

import streamlit as st
from typing import List, Dict
import random
import time
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()  # loads .env file
try:
    import openai
except ImportError:
    openai = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.warning("⚠️ OPENAI_API_KEY not found. AI Mode will fallback to Test Mode.")

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
        symptoms = PRESET_CASES[preset]
        st.text_area("Patient symptoms", value=symptoms, height=100)
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
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    text = response['choices'][0]['message']['content']
                    result = {"differential": [text], "explanation": [text], "recommended_tests": [text]}

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
# Psychiatrist Chatbot with Memory (Hidden Preloaded Lines)
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
            reply = "[Test Mode] It sounds like you are feeling anxious. Try deep breathing, journaling, and consult a mental health professional if persistent."
        else:
            # Include all messages in memory, hidden or visible, for AI context
            context_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.chat_history
            ] + [{"role": "user", "content": user_input}]
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=context_messages
            )
            reply = response['choices'][0]['message']['content']

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

st.success("NeuroSync fully enhanced with polished UI, expanded flashcards, and real-time psychiatrist chatbot ready!")
