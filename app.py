import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from streamlit_lottie import st_lottie
import requests

# ✅ Load models
model = pickle.load(open("flight_rf.pkl", "rb"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.idx")
with open("flight_docs.txt", encoding="utf-8") as f:
    documents = f.read().split('\n\n')
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# 🎬 Load Lottie animations
def load_lottie_url(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_airplane = load_lottie_url("https://lottie.host/6f9eec3e-6894-44e8-b94e-2b2600b94c1a/1RrgPf63VJ.json")
lottie_footer = load_lottie_url("https://lottie.host/5dd2ab42-b262-4e03-8250-1b6e8eeb4a57/1YJ3wK94U6.json")

# 🌙 / ☀️ Theme toggle (fake toggle just for demo)
mode = st.sidebar.radio("🌓 Theme", ["Light", "Dark"])

# 🎨 Fancy CSS styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@500&display=swap');
html, body, [class*="css"] {{
    font-family: 'Roboto Slab', serif;
    background: linear-gradient(120deg, {'#ffffff, #e0e0e0' if mode=='Light' else '#1a2a6c, #b21f1f, #fdbb2d'});
    color: {'#000' if mode=='Light' else '#fff'};
}}
h1 {{
    text-align: center;
    font-size: 36px;
    color: #00ffe5;
    text-shadow: 2px 2px 4px #000;
    animation: fadeIn 2s ease-in;
}}
.stButton>button {{
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.4em 1em;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 0 10px #00c6ff;
}}
.stButton>button:hover {{
    transform: scale(1.05);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}}
@keyframes fadeIn {{
    from {{opacity:0;}}
    to {{opacity:1;}}
}}
</style>
""", unsafe_allow_html=True)

# ✨ Splash screen (shows only once)
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = True
    st.markdown("<h1>🚀 Welcome to Flight AI App!</h1>", unsafe_allow_html=True)
    if lottie_airplane: st_lottie(lottie_airplane, speed=1, height=200)
    st.stop()

def predict():
    st.sidebar.title("🛫 ✈️ Flight AI App")
    st.sidebar.write("🔍 Predict flight prices + ask about project")
    if lottie_airplane:
        st.sidebar.markdown("---")
        st_lottie(lottie_airplane, speed=1, height=150)

    # Sidebar: LLM + RAG
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Ask about this project:")
    user_question = st.sidebar.text_input("Type your question:")
    if user_question:
        question_embedding = embedder.encode([user_question])
        scores, indices = index.search(np.array(question_embedding), k=1)
        retrieved_text = documents[indices[0][0]]
        prompt = f"Question: {user_question}\nContext: {retrieved_text}\nAnswer:"
        answer = generator(prompt, max_length=100)[0]['generated_text']
        st.sidebar.success(f"📚 {answer}")

    # Title
    st.markdown("<h1>✨ Flight Price Predictor with 🤖 RAG + LLM</h1>", unsafe_allow_html=True)

    # Inputs
    date_dep = st.date_input("📅 Select Departure Date")
    Journey_day, Journey_month = date_dep.day, date_dep.month
    Dep_hour = st.number_input("🕑 Departure Hour (0–23)", 0, 23, 0)
    Dep_min = st.number_input("🕑 Departure Minute (0–59)", 0, 59, 0)
    Arrival_hour = st.number_input("🛬 Arrival Hour (0–23)", 0, 23, 0)
    Arrival_min = st.number_input("🛬 Arrival Minute (0–59)", 0, 59, 0)
    dur_hour, dur_min = abs(Arrival_hour - Dep_hour), abs(Arrival_min - Dep_min)
    Total_stops = st.selectbox("✈️ Number of stops", [0,1,2,3,4])

    airline = st.selectbox("🛫 Airline", [
        "Jet Airways", "IndiGo", "Air India", "Multiple carriers", "SpiceJet",
        "Vistara", "Air Asia", "GoAir", "Multiple carriers Premium economy",
        "Jet Airways Business", "Vistara Premium economy", "Trujet"
    ])
    Jet_Airways=IndiGo=Air_India=Multiple_carriers=SpiceJet=Vistara=Air_Asia=GoAir=Multiple_carriers_Premium_economy=Jet_Airways_Business=Vistara_Premium_economy=Trujet=0
    if airline=='Jet Airways': Jet_Airways=1
    elif airline=='IndiGo': IndiGo=1
    elif airline=='Air India': Air_India=1
    elif airline=='Multiple carriers': Multiple_carriers=1
    elif airline=='SpiceJet': SpiceJet=1
    elif airline=='Vistara': Vistara=1
    elif airline=='Air Asia': Air_Asia=1
    elif airline=='GoAir': GoAir=1
    elif airline=='Multiple carriers Premium economy': Multiple_carriers_Premium_economy=1
    elif airline=='Jet Airways Business': Jet_Airways_Business=1
    elif airline=='Vistara Premium economy': Vistara_Premium_economy=1
    elif airline=='Trujet': Trujet=1

    Source = st.selectbox("🏙 Source city", ["Delhi", "Kolkata", "Mumbai", "Chennai"])
    s_Delhi=s_Kolkata=s_Mumbai=s_Chennai=0
    if Source=='Delhi': s_Delhi=1
    elif Source=='Kolkata': s_Kolkata=1
    elif Source=='Mumbai': s_Mumbai=1
    elif Source=='Chennai': s_Chennai=1

    destination = st.selectbox("🏙 Destination city", ["Cochin","Delhi","New_Delhi","Hyderabad","Kolkata"])
    d_Cochin=d_Delhi=d_New_Delhi=d_Hyderabad=d_Kolkata=0
    if destination=='Cochin': d_Cochin=1
    elif destination=='Delhi': d_Delhi=1
    elif destination=='New_Delhi': d_New_Delhi=1
    elif destination=='Hyderabad': d_Hyderabad=1
    elif destination=='Kolkata': d_Kolkata=1

    if st.button("🚀 Predict Flight Price"):
        pred = model.predict([[Total_stops, Journey_day, Journey_month, Dep_hour, Dep_min,
            Arrival_hour, Arrival_min, dur_hour, dur_min,
            Air_India, GoAir, IndiGo, Jet_Airways, Jet_Airways_Business,
            Multiple_carriers, Multiple_carriers_Premium_economy, SpiceJet, Trujet,
            Vistara, Vistara_Premium_economy, s_Chennai, s_Delhi, s_Kolkata, s_Mumbai,
            d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata, d_New_Delhi]])
        st.success(f"✅ Predicted Flight Price: ₹ {round(pred[0],2)} 🎉")

    # Footer animation
    if lottie_footer:
        st_lottie(lottie_footer, speed=1, height=100)
    st.markdown("<p style='text-align:center; color:#eee;'>✨ Stylish UI + AI by SAURAV PATTNAIK</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    predict()