import streamlit as st
import pandas as pd
import joblib
import os

# --- APP ASSETS LOAD KARNA ---
@st.cache_resource
def load_assets():
    if os.path.exists('movie_expert_model.pkl') and os.path.exists('expert_encoders.pkl'):
        model = joblib.load('movie_expert_model.pkl')
        encoders = joblib.load('expert_encoders.pkl')
        return model, encoders
    return None, None

model, encoders = load_assets()

# --- UI INTERFACE ---
st.title("Movie Success Predictor")
st.write("Enter the movie details below to get a box office prediction.")

if model is None:
    st.error("Model files not found. Please run the training script first.")
else:
    # --- INPUT SECTION ---
    # Simple vertical layout without complex columns or colors
    
    st.header("Movie Details")
    genre = st.selectbox("Genre", sorted(encoders['Genre'].classes_))
    actor = st.selectbox("Actor", sorted(encoders['Actor'].classes_))
    director = st.selectbox("Director", sorted(encoders['Director'].classes_))
    country = st.selectbox("Country", sorted(encoders['Country'].classes_))
    
    st.divider()
    
    st.header("Parameters")
    budget = st.slider("Budget (Million USD)", 1, 500, 50)
    screens = st.number_input("Number of Screens", min_value=100, max_value=10000, value=1200)
    weeks = st.slider("Weeks in Theater", 1, 15, 4)
    stability = st.selectbox("Screen Stability", sorted(encoders['Screen_Stability'].classes_))
    price = st.slider("Ticket Price ($)", 5, 100, 15)
    shows = st.slider("Shows per Day", 1, 24, 8)

    st.divider()

    # --- PREDICTION ACTION ---
    if st.button("Predict Result"):
        # Data prepare karna
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

        # Prediction
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Calculation logic
        stab_map = {'Drop Fast': 0.45, 'Slow Drop': 0.85, 'Stable': 1.25, 'Growing': 1.9}
        est_rev = (screens * shows * 35 * price * weeks * stab_map[stability]) / 1000000
        
        # Multipliers
        if 'Animation' in genre or '(Voice)' in actor:
            est_rev *= 1.25
            
        profit_loss = est_rev - budget
        rating = ((weeks * 0.4) + (2.5 if prediction == 1 else 0) + 3.5)
        rating = min(max(rating, 1.0), 10.0)

        # --- RESULTS (SIMPLE DISPLAY) ---
        st.subheader("Analysis Results")
        
        if prediction == 1:
            st.success(f"Status: HIT (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"Status: FLOP (Confidence: {(1-prob)*100:.1f}%)")

        # Stats dikhane ke liye simple columns
        col1, col2 = st.columns(2)
        col1.metric("Est. Profit/Loss", f"{profit_loss:+.2f} M$")
        col2.metric("Audience Rating", f"{rating:.1f} / 10")

        st.info(f"Estimated Lifetime Revenue: ${est_rev:.2f} Million")

st.divider()
st.caption("Minimalist AI Predictor | Trained on 5,000 unique records.")