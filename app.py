import streamlit as st
import pickle
import pdfplumber
import re
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load Model
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------
# Text Cleaning
# -------------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

# -------------------------
# Custom CSS Styling
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stButton>button {
    background-color: #00C853;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #00A844;
}
.prediction-box {
    background-color: #1B5E20;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("<h1 style='text-align:center;'>📄 AI Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Upload your resume and get instant job category prediction</p>", unsafe_allow_html=True)

st.divider()

# -------------------------
# Layout Columns
# -------------------------
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])

with col2:
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            resume_text = ""
            for page in pdf.pages:
                resume_text += page.extract_text()

        st.subheader("Resume Preview")
        st.text_area("Preview", resume_text[:1500], height=250, label_visibility="collapsed")

        if st.button("🚀 Analyze Resume"):

            cleaned = clean_text(resume_text)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2 style="color:white;">Predicted Category</h2>
                    <h1 style="color:white;">{prediction}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # If probability available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vectorized)[0]

                proba_df = pd.DataFrame({
                    "Category": model.classes_,
                    "Confidence": probabilities
                }).sort_values(by="Confidence", ascending=False)

                st.subheader("Top Confidence Scores")

                fig, ax = plt.subplots(figsize=(8,4))
                ax.barh(proba_df["Category"][:5], proba_df["Confidence"][:5])
                ax.invert_yaxis()
                st.pyplot(fig)

st.divider()
st.markdown("<p style='text-align:center;color:gray;'>Built with ❤️ using Machine Learning & Streamlit</p>", unsafe_allow_html=True)