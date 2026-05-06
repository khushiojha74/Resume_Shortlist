# Smart Resume Screening System (AI-Powered)

An AI-powered resume screening system that extracts skills, matches them against job descriptions using semantic embeddings, and provides detailed assessments.

## Features

- **Resume Parsing**: Extracts text from PDF and DOCX files
- **Skill Extraction**: Identifies relevant skills from resumes
- **Semantic Matching**: Uses SentenceTransformers for embedding-based similarity scoring
- **AI Assessment**: Generates recruiter-style explanations using Cohere AI
- **REST API**: FastAPI endpoint for programmatic access

## Requirements

- Python 3.8+
- Cohere API key (for AI explanations)

## Setup

1. **Clone the repository**
   ```bash
   git clone -
   cd Resume_Shortlisting
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env`
   - Add your Cohere API key:
     ```
     COHERE_API_KEY=your_actual_api_key_here
     ```"Currently added for testing purpose"

## How to Run

```bash
python semantic_resume_matcher.py
```

This will process resumes in the `resumes/` folder against `jd.txt` and print results to console.

### Option 2: Run as API

with uvicorn:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

#### API Usage

**Endpoint**: `POST /screen-resumes`

**Request Body**:
```json
{
  "jd_text": "Job description text here...",
  "resume_paths": [
    "path/to/resume1.pdf",
    "path/to/resume2.docx"
  ]
}
```

**Response**:
```json
[
  {
    "resume_name": "resume1.pdf",
    "matching_score": 85,
    "matched_skills": ["python", "fastapi", "docker"],
    "missing_skills": ["aws", "kubernetes"],
    "explanation": "Strengths:\n- Strong Python and API development skills\n\nGaps:\n- Cloud deployment experience needed\n\nSummary:\n- Excellent technical foundation"
  }
]
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Approach Explanation

### 1. Text Extraction
- Uses `pdfplumber` for PDF parsing
- Uses `python-docx` for DOCX parsing
- Handles both file types seamlessly

### 2. Skill Extraction
- Cleans and tokenizes text
- Removes common stopwords
- Extracts keywords from job description
- Matches against resume content using regex

### 3. Semantic Similarity
- Uses `SentenceTransformer` with `all-MiniLM-L6-v2` model
- Computes cosine similarity between JD and resume embeddings
- Provides 0-100 match score

### 4. AI Assessment
- Leverages Cohere's `command-r-plus` model
- Generates structured explanations in Strengths/Gaps/Summary format
- Falls back to basic assessment if API unavailable

### 5. API Design
- Built with FastAPI for high performance
- Pydantic models for request/response validation
- Proper error handling and HTTP status codes

## Project Structure

```
Resume_Shortlisting/
├── app.py                 # FastAPI application
├── semantic_resume_matcher.py  # Core matching logic
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── jd.txt               # Sample job description
└── resumes/             # Resume files directory
```

## Security Notes

- Never commit `.env` file with real API keys
- Use environment variables for sensitive data
- The `.gitignore` prevents accidental commits
