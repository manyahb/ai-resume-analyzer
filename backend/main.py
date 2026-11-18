from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze, extract_text

app = FastAPI(title="AI Resume Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_BYTES = 5 * 1024 * 1024


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    data = await resume.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 5 MB.")
    if len(job_description.strip()) < 30:
        raise HTTPException(400, "Paste a longer job description (at least a few lines).")
    try:
        text = extract_text(resume.filename, data)
        return analyze(text, job_description)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(400, "Couldn't process that file. Try another PDF.")
