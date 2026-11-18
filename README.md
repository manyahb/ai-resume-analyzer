# AI Resume Analyzer (Vite + FastAPI)

Scores a resume against a job description with TF-IDF and cosine similarity, then reports skill gaps,
keyword coverage, a resume summary, and highlighted matches. Includes a downloadable report.

## Run it

Backend (Python 3.9+):

    cd backend
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Frontend (Node 18+), in a second terminal:

    cd frontend
    npm install
    npm run dev

Open http://localhost:5173. Vite proxies `/api` to the backend on port 8000.

## How scoring works (0-100)

- 35% text similarity: stemmed TF-IDF (1-2 grams) cosine between resume and job description
- 30% skill match: skills in the job description that also appear on the resume
- 20% keyword coverage: top job-description terms found in the resume
- 15% structure: contact, education, experience, projects, skills, measurable results

Add skills to `SKILLS` in `backend/analyzer.py` to extend detection.
