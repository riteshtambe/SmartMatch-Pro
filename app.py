
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import pdfplumber
from fpdf import FPDF
import io
import re

# ------------------------ CLEAN TEXT ------------------------
def clean_text(text):
    replacements = {
        "–": "-", "—": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "•": "-", "→": "->"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Remove any non-Latin-1 characters (like emojis)
    text = re.sub(r'[^\x00-\xFF]+', '', text)
    return text

# # ------------------------ PDF CLASS ------------------------

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, "SmartMatch Resume-JD Report", ln=True, align="C")
        self.ln(5)
        self.set_draw_color(0, 0, 0)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def add_section(self, title, content):
        self.set_font("Arial", 'B', 13)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", '', 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 8, content)
        self.ln(3)

    def add_list_section(self, title, items):
        self.set_font("Arial", 'B', 13)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", '', 11)
        self.set_text_color(50, 50, 50)
        if items:
            for i, item in enumerate(items, 1):
                self.multi_cell(0, 8, f"{i}. {item}")
        else:
            self.cell(0, 10, "None", ln=True)
        self.ln(3)

def generate_pdf(score, matched, missing, resume_text, jd_text):
    pdf = PDF()
    pdf.add_page()

    # Match score section
    feedback = (
        "Strong Match! Your resume aligns well with the job description." if score > 75 else
        "Moderate Match. Consider updating your resume to match more skills." if score > 50 else
        "Weak Match. Add more relevant skills to your resume to increase the match score."
    )
    pdf.add_section("Match Score Summary", f"Match Score: {score}%\n\nFeedback: {feedback}")

    # Skills overview
    pdf.add_list_section("Matched Skills", matched)
    pdf.add_list_section("Missing Skills", missing)

    # Skill suggestions (missing skills repurposed)
    if missing:
        pdf.add_list_section("Skills to Add to Improve Resume", missing)

    # Job description snippet (keep this only)
    pdf.add_section("Job Description Snippet", jd_text[:700] + "..." if len(jd_text) > 700 else jd_text)

    # Output
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return io.BytesIO(pdf_bytes)


# ------------------------ LOAD MODELS ------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedder)

# ------------------------ STREAMLIT UI ------------------------
st.set_page_config(page_title="SmartMatch AI", layout="centered")

st.sidebar.image("logo.png", width=180)
st.sidebar.title("🔍 SmartMatch AI")
st.sidebar.markdown("Upload your resume and paste a job description to check compatibility and get feedback.")

st.title("🧠 Resume ↔ JD Matcher with Skill Highlighter")
st.markdown("Compare resumes with job descriptions, calculate match score, and view matching skills.")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
resume_text = ""

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            resume_text += page.extract_text() or ""
    st.success("✅ Resume uploaded successfully!")

job_description = st.text_area("📝 Paste Job Description here", height=200)

if st.button("🔎 Match Now"):
    if resume_text.strip() == "" or job_description.strip() == "":
        st.error("❗ Please upload a resume and paste a job description.")
    else:
        with st.spinner("🔍 Processing..."):
            # Clean text
            resume_text_cleaned = clean_text(resume_text)
            job_description_cleaned = clean_text(job_description)

            # Embedding-based score
            embeddings = embedder.encode([resume_text_cleaned, job_description_cleaned])
            score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            score_pct = round(score * 100, 2)

            # Keyword extraction
            resume_keywords = kw_model.extract_keywords(resume_text_cleaned, top_n=20, stop_words='english')
            jd_keywords = kw_model.extract_keywords(job_description_cleaned, top_n=20, stop_words='english')

            resume_kw_set = set([kw[0].lower() for kw in resume_keywords])
            jd_kw_set = set([kw[0].lower() for kw in jd_keywords])

            matched = sorted(list(resume_kw_set & jd_kw_set))
            missing = sorted(list(jd_kw_set - resume_kw_set))

            # Show score
            st.markdown("---")
            st.subheader("📊 Match Score")
            if score_pct > 75:
                st.success(f"✅ Strong Match: {score_pct}%")
            elif score_pct > 50:
                st.warning(f"⚠️ Moderate Match: {score_pct}%")
            else:
                st.error(f"❌ Weak Match: {score_pct}%")

            # Skills Display
            st.markdown("### 🎯 Matched Skills")
            st.success(", ".join(matched) if matched else "None")

            st.markdown("### ❌ Missing Skills")
            st.info(", ".join(missing) if missing else "None")

            # CSV Download
            result_df = pd.DataFrame({
                "Match Score (%)": [score_pct],
                "Matched Skills": [", ".join(matched)],
                "Missing Skills": [", ".join(missing)],
                "JD Snippet": [job_description_cleaned[:150]],
                "Resume Snippet": [resume_text_cleaned[:150]],
            })

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "match_result.csv", "text/csv")

            # PDF Download
            pdf_bytes = generate_pdf(score_pct, matched, missing, resume_text_cleaned, job_description_cleaned)
            st.download_button("📄 Download PDF Report", pdf_bytes, "match_report.pdf", mime="application/pdf")
