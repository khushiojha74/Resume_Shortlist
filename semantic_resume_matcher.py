import os
import re
import pdfplumber
from docx import Document
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading environment from: {dotenv_path}")

print("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

# Initialize Cohere client from environment
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise EnvironmentError(
        "COHERE_API_KEY is missing. Add it to a .env file or your environment variables."
    )
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        print(f"[ERROR] PDF ({pdf_path}): {e}")
    return text


def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"[ERROR] DOCX ({docx_path}): {e}")
    return text


def extract_resume_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        print(f"[SKIP] Unsupported file: {file_path}")
        return ""

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================================================
# KEYWORD EXTRACTION
# =========================================================

STOPWORDS = {
    "the", "and", "for", "with", "are", "you",
    "our", "your", "will", "have", "has",
    "this", "that", "from", "their", "they",
    "who", "all", "can", "using"
}

def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]+\b", text)

    keywords = [
        w for w in words
        if len(w) > 2 and w not in STOPWORDS
    ]

    return sorted(set(keywords))

# =========================================================
# SKILL MATCHING
# =========================================================

def find_skill_matches(jd_keywords, resume_text):
    matched, missing = [], []

    for keyword in jd_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", resume_text):
            matched.append(keyword)
        else:
            missing.append(keyword)

    return matched, missing

# =========================================================
# EXPLANATION GENERATION (COHERE)
# =========================================================

def generate_explanation(matched_skills, missing_skills, score):
    """Generate explanation using Cohere Chat API"""
    try:
        matched_str = ", ".join(matched_skills[:5]) if matched_skills else "None"
        missing_str = ", ".join(missing_skills[:5]) if missing_skills else "None"
        
        prompt = f"""Act as an AI recruiter.

Based on:
- Match Score: {score}/100
- Matched Skills: {matched_str}
- Missing Skills: {missing_str}

Return output in this format:

Strengths:
- ...

Gaps:
- ...

Summary:
- ...

Keep everything very short (2-3 lines total)."""
        
        response = co.chat(
            model="command",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        explanation = response.message.content[0].text.strip() if response.message.content else "Assessment generated."
        return explanation[:500]  # Cap at 500 chars for more content
    except Exception as e:
        print(f"[ERROR] Cohere explanation: {e}")
        matched_count = len(matched_skills)
        missing_count = len(missing_skills)
        return f"Strengths: Candidate has {matched_count} relevant skills. Gaps: {missing_count} skills to develop. Summary: Good foundation with room for upskilling."

# =========================================================
# SEMANTIC SIMILARITY
# =========================================================

def calculate_semantic_score(jd_text, resume_text):
    try:
        embeddings = model.encode([jd_text, resume_text])
        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return max(1, min(round(similarity * 100), 100))
    except Exception as e:
        print(f"[ERROR] Embedding: {e}")
        return 0

# =========================================================
# MATCH FUNCTION
# =========================================================

def match_resume_with_jd(jd_text, resume_text):
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = calculate_semantic_score(jd_clean, resume_clean)

    jd_keywords = extract_keywords(jd_clean)

    matched, missing = find_skill_matches(
        jd_keywords,
        resume_clean
    )
    
    # Generate Cohere explanation
    explanation = generate_explanation(matched, missing, score)

    return {
        "matching_score": score,
        "matched_skills": matched[:20],
        "missing_skills": missing[:20],
        "explanation": explanation,
    }

# =========================================================
# API HELPER FUNCTION
# =========================================================

def process_resumes_from_texts(jd_text, resume_paths):
    """Process resumes from JD text and list of resume file paths (for API use)"""
    results = []

    for file_path in resume_paths:
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing: {os.path.basename(file_path)}")

        resume_text = extract_resume_text(file_path)

        if not resume_text.strip():
            print("[SKIP] Empty content")
            continue

        result = match_resume_with_jd(jd_text, resume_text)
        result["resume_name"] = os.path.basename(file_path)

        results.append(result)

    return sorted(results, key=lambda x: x["matching_score"], reverse=True)

# =========================================================
# PROCESS RESUMES (for CLI)
# =========================================================

def process_resumes(jd_file, resumes_folder):
    if not os.path.exists(jd_file):
        raise FileNotFoundError("JD file not found")

    with open(jd_file, "r", encoding="utf-8") as f:
        jd_text = f.read()

    results = []

    for file_name in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, file_name)

        if not os.path.isfile(file_path):
            continue
        resume_text = extract_resume_text(file_path)

        if not resume_text.strip():
            print("[SKIP] Empty content")
            continue

        result = match_resume_with_jd(jd_text, resume_text)
        result["resume_name"] = file_name

        results.append(result)

    return sorted(results, key=lambda x: x["matching_score"], reverse=True)



# =========================================================
# MAIN
# =========================================================

# =========================================================
# MAIN (for CLI usage)
# =========================================================

if __name__ == "__main__":

    JD_FILE = "jd.txt"
    RESUMES_FOLDER = "resumes"

    results = process_resumes(JD_FILE, RESUMES_FOLDER)

    print("\n" + "=" * 70)
    print("AI SEMANTIC RESUME SCREENING RESULTS")
    print("=" * 70)

    for res in results:
        print(f"\nResume: {res['resume_name']}")
        print(f"Score: {res['matching_score']}/100")
        print(f"Matched: {', '.join(res['matched_skills'])}")
        print(f"Missing: {', '.join(res['missing_skills'])}")
        print(f"\nAssessment:")
        print(res.get('explanation', 'N/A'))
        print("-" * 70)