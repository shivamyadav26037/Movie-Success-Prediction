import streamlit as st
import pandas as pd
import joblib
import os

# --- ASSETS LOAD KARNA ---
@st.cache_resource
def load_assets():
    if os.path.exists('movie_expert_model.pkl') and os.path.exists('expert_encoders.pkl'):
        model = joblib.load('movie_expert_model.pkl')
        encoders = joblib.load('expert_encoders.pkl')
        return model, encoders
    return None, None

model, encoders = load_assets()

# --- APP UI (SIMPLE & MINIMAL) ---
st.set_page_config(page_title="Movie Success Predictor", layout="centered")

st.title("Movie Success Predictor")
st.write("Fill in the details below to analyze movie performance.")

if model is None:
    st.error("Model files not found. Please run the training script (trainer.py) first.")
else:
    # --- INPUT SECTION ---
    st.header("1. Production Details")
    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Genre", sorted(encoders['Genre'].classes_))
        actor = st.selectbox("Actor", sorted(encoders['Actor'].classes_))
        country = st.selectbox("Country", sorted(encoders['Country'].classes_))
    with col2:
        director = st.selectbox("Director", sorted(encoders['Director'].classes_))
        stability = st.selectbox("Screen Stability", sorted(encoders['Screen_Stability'].classes_))

    st.header("2. Financials & Release")
    budget = st.slider("Budget (Million USD)", 1, 500, 50)
    screens = st.number_input("Number of Screens", min_value=100, max_value=10000, value=1200)
    weeks = st.slider("Weeks in Theater", 1, 15, 4)
    price = st.slider("Ticket Price ($)", 5, 100, 15)
    shows = st.slider("Shows per Day", 1, 24, 8)

    st.divider()

    # --- PREDICTION ACTION ---
    if st.button("Predict Result", use_container_width=True):
        # Data prepare karna model ke liye
        input_data = pd.DataFrame([{
            'Genre': encoders['Genre'].transform([genre])[0],
            'Actor': encoders['Actor'].transform([actor])[0],
            'Director': encoders['Director'].transform([director])[0],
            'Country': encoders['Country'].transform([country])[0],
            'Budget_MUSD': budget,
            'Screens': screens,
            'Weeks': weeks,
            'Screen_Stability': encoders['Screen_Stability'].transform([stability])[0],
            'Ticket_Price': price,
            'Shows_per_Day': shows
        }])

        # Model Prediction (for Confidence)
        prob = model.predict_proba(input_data)[0][1]

        # Financial Calculation (Consistent with Training Logic)
        stab_map = {'Drop Fast': 0.45, 'Slow Drop': 0.85, 'Stable': 1.25, 'Growing': 1.9}
        # Base Calculation
        est_rev = (screens * shows * 35 * price * weeks * stab_map[stability]) / 1000000
        
        # Applying Multipliers
        if 'Animation' in genre or '(Voice)' in actor:
            est_rev *= 1.25
        if weeks > 6:
            est_rev *= 1.2
            
        profit_loss = est_rev - budget
        
        # Final Status based on Financial Reality
        # Agar Revenue >= Budget, toh HIT. Warna FLOP.
        is_hit = profit_loss >= 0
        
        # Rating Prediction Logic
        rating = ((weeks * 0.4) + (2.5 if is_hit else 0) + 3.5)
        rating = min(max(rating, 1.0), 10.0)

        # --- RESULTS DISPLAY ---
        st.subheader("Analysis Results")
        
        if is_hit:
            st.success(f"Status: HIT (Model Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"Status: FLOP (Model Confidence: {(1-prob)*100:.1f}%)")

        col1, col2 = st.columns(2)
        # Green color for profit, Red for loss
        col1.metric("Est. Profit/Loss", f"{profit_loss:+.2f} M$", delta=f"{profit_loss:.2f} M$")
        col2.metric("Audience Rating", f"{rating:.1f} / 10")

        st.info(f"Estimated Lifetime Revenue: **${est_rev:.2f} Million**")

