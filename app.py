import streamlit as st
import joblib

# Load model
model = joblib.load("model/model.pkl")

# UI
st.title("📧 CRM Text Classifier")
st.subheader("Classify a customer message into: Complaint, Query, Feedback, or Spam")

# Input box
text_input = st.text_area("✍️ Enter customer message:")

if st.button("🔍 Classify"):
    if text_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([text_input])[0]
        st.success(f"✅ Predicted category: **{prediction.upper()}**")
