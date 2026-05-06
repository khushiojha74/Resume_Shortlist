from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from semantic_resume_matcher import process_resumes_from_texts

app = FastAPI(title="Resume Screening API", description="AI-powered resume screening system")

class ResumeScreeningRequest(BaseModel):
    jd_text: str
    resume_paths: List[str]

class ScreeningResult(BaseModel):
    resume_name: str
    matching_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    explanation: str

@app.post("/screen-resumes", response_model=List[ScreeningResult])
async def screen_resumes(request: ResumeScreeningRequest):
    """
    Screen resumes against a job description.

    - **jd_text**: The job description text
    - **resume_paths**: List of file paths to resume files (PDF/DOCX)
    """
    try:
        # Validate that all resume files exist
        for path in request.resume_paths:
            if not os.path.exists(path):
                raise HTTPException(status_code=400, detail=f"Resume file not found: {path}")

        # Process resumes
        results = process_resumes_from_texts(request.jd_text, request.resume_paths)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)