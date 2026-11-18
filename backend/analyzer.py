"""NLP engine: TF-IDF + cosine similarity, skill-gap, summarization, ATS score."""
import io
import re
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _load_stopwords():
    stop = set(ENGLISH_STOP_WORDS)
    try:
        import nltk
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        stop |= set(stopwords.words("english"))
    except Exception:
        pass  # sklearn's list is a fine fallback when offline
    return stop


STOP = _load_stopwords()
STEM = PorterStemmer()

# Job-posting filler that says nothing about fit
FILLER = set("""need needs use used using work works working build building built team teams experience experienced
year years role roles job will can able must strong good great ability looking join help including include
required requirements preferred plus within across new etc well also like make making want knowledge understanding
responsibilities responsible candidate company position skills skill engineer developer""".split())

SKILLS = {
    "Languages": ["python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust", "kotlin", "swift",
                  "php", "ruby", "scala", "r", "matlab", "sql", "bash", "html", "css"],
    "ML / AI": ["machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
                "llm", "llms", "generative ai", "transformers", "langchain", "rag", "tensorflow", "pytorch", "keras",
                "scikit-learn", "sklearn", "nltk", "spacy", "hugging face", "opencv", "xgboost", "tf-idf",
                "reinforcement learning", "data science", "feature engineering", "prompt engineering"],
    "Data": ["pandas", "numpy", "matplotlib", "seaborn", "power bi", "tableau", "excel", "spark", "hadoop",
             "etl", "data analysis", "data visualization", "statistics", "airflow", "kafka"],
    "Backend & Web": ["fastapi", "flask", "django", "node.js", "express", "react", "vite", "next.js", "angular",
                      "vue", "streamlit", "rest", "restful", "graphql", "microservices", "api", "websocket",
                      "tailwind", "redux", "spring boot"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "terraform", "linux",
                       "git", "github", "github actions", "ec2", "s3", "lambda", "nginx", "devops"],
    "Databases": ["mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite", "firebase", "dynamodb",
                  "elasticsearch", "oracle", "nosql"],
    "Practices": ["agile", "scrum", "oop", "data structures", "algorithms", "system design", "unit testing",
                  "pytest", "jira", "problem solving", "communication", "leadership", "teamwork"],
}
ALL_SKILLS = {s: cat for cat, items in SKILLS.items() for s in items}
ALIASES = {"llms": "llm", "sklearn": "scikit-learn", "postgres": "postgresql", "restful": "rest",
           "natural language processing": "nlp"}

SECTIONS = {
    "Contact info": r"@[\w.-]+\.\w+|\+?\d[\d\s().-]{8,}\d",
    "Education": r"\beducation\b|\bb\.?e\.?\b|\bb\.?tech\b|\bbachelor|\bmaster|\bcgpa|\buniversity|\bcollege",
    "Experience / Internships": r"\bexperience\b|\bintern(ship)?\b|\bemployment\b|\bwork history\b",
    "Projects": r"\bprojects?\b",
    "Skills": r"\bskills?\b|\btechnical skills\b|\btechnologies\b",
    "Measurable results": r"\d+\s?%|\b\d{2,}\+?\s?(users|records|requests|ms|x)\b|\bimproved\b|\breduced\b|\bincreased\b",
}


def extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected.")
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Upload a PDF or TXT resume.")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 40:
        raise ValueError("Couldn't read text from this file. If it's a scanned PDF, export a text-based one.")
    return text


def _tokens(text: str):
    return [t.rstrip(".") for t in re.findall(r"[a-z][a-z0-9+#.\-]*", text.lower())]


def preprocess(text: str) -> str:
    return " ".join(STEM.stem(t) for t in _tokens(text) if t not in STOP and len(t) > 1)


def find_skills(text: str) -> set:
    low = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        pat = r"(?<![\w+#.])" + re.escape(skill) + r"(?![\w+#]|\.\w)"
        if re.search(pat, low):
            found.add(ALIASES.get(skill, skill))
    return found


def keyword_coverage(resume: str, jd: str, top_n: int = 25):
    """Top job-description terms (by frequency) and which ones the resume contains."""
    counts = Counter(t for t in _tokens(jd) if t not in STOP and t not in FILLER and len(t) > 2 and not t.isdigit())
    stem_groups = {}
    for word, c in counts.items():
        stem_groups.setdefault(STEM.stem(word), []).append((c, word))
    ranked = sorted(stem_groups.items(), key=lambda kv: -sum(c for c, _ in kv[1]))[:top_n]
    resume_stems = {STEM.stem(t) for t in _tokens(resume)}
    matched, missing = [], []
    for stem, variants in ranked:
        word = max(variants)[1]
        (matched if stem in resume_stems else missing).append(word)
    return matched, missing


def summarize(text: str, n: int = 4) -> list:
    lines = [s.strip(" •-–*\t") for s in re.split(r"\n|(?<=[.!?])\s+", text)]
    lines = [s for s in lines if 45 <= len(s) <= 300 and len(s.split()) >= 7]
    if not lines:
        return []
    freq = Counter(STEM.stem(t) for t in _tokens(text) if t not in STOP and len(t) > 2)
    top = max(freq.values()) if freq else 1
    scored = []
    for i, s in enumerate(lines):
        toks = [STEM.stem(t) for t in _tokens(s) if t not in STOP and len(t) > 2]
        score = sum(freq[t] / top for t in toks) / (len(toks) ** 0.5 + 1)
        scored.append((score, i, s))
    best = sorted(scored, reverse=True)[:n]
    return [s for _, _, s in sorted(best, key=lambda x: x[1])]


def analyze(resume_text: str, jd_text: str) -> dict:
    r_clean, j_clean = preprocess(resume_text), preprocess(jd_text)
    if not j_clean:
        raise ValueError("Job description has too little text to analyze.")

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf = vec.fit_transform([r_clean, j_clean])
    cosine = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    sim_score = float(np.clip(cosine / 0.6, 0, 1))  # ~0.6 raw cosine is an excellent real-world match

    r_sk, j_sk = find_skills(resume_text), find_skills(jd_text)
    matched_sk, missing_sk = sorted(r_sk & j_sk), sorted(j_sk - r_sk)
    extra_sk = sorted(r_sk - j_sk)
    skill_score = len(matched_sk) / len(j_sk) if j_sk else sim_score

    kw_matched, kw_missing = keyword_coverage(resume_text, jd_text)
    kw_total = len(kw_matched) + len(kw_missing)
    kw_score = len(kw_matched) / kw_total if kw_total else 0

    sec = {name: bool(re.search(p, resume_text, re.I)) for name, p in SECTIONS.items()}
    struct_score = sum(sec.values()) / len(sec)

    ats = 100 * (0.35 * sim_score + 0.30 * skill_score + 0.20 * kw_score + 0.15 * struct_score)

    tips = []
    if missing_sk:
        tips.append("Add these skills if you genuinely have them: " + ", ".join(missing_sk[:8]) + ".")
    if kw_missing:
        tips.append("Work these job-description terms into your bullets: " + ", ".join(kw_missing[:8]) + ".")
    for name, ok in sec.items():
        if not ok:
            tips.append(f"Missing section or signal: {name}.")
    if not sec["Measurable results"]:
        tips.append("Quantify impact (for example, 'cut latency by 30%') to strengthen your bullets.")
    if len(resume_text.split()) < 150:
        tips.append("Your resume looks short. Add detail to projects and experience.")
    if not tips:
        tips.append("Strong match. Tailor the summary line to the role and you're set.")

    highlight = sorted(set(matched_sk) | set(kw_matched), key=len, reverse=True)
    verdict = "Strong match" if ats >= 75 else "Good match" if ats >= 55 else "Partial match" if ats >= 35 else "Weak match"
    return {
        "ats_score": round(ats, 1),
        "verdict": verdict,
        "cosine_similarity": round(cosine * 100, 1),
        "breakdown": {
            "Text similarity": round(sim_score * 100),
            "Skill match": round(skill_score * 100),
            "Keyword coverage": round(kw_score * 100),
            "Resume structure": round(struct_score * 100),
        },
        "skills": {
            "matched": [{"name": s, "category": ALL_SKILLS.get(s, "Other")} for s in matched_sk],
            "missing": [{"name": s, "category": ALL_SKILLS.get(s, "Other")} for s in missing_sk],
            "extra": extra_sk,
        },
        "keywords": {"matched": kw_matched, "missing": kw_missing},
        "sections": sec,
        "summary": summarize(resume_text),
        "suggestions": tips,
        "highlight_terms": highlight,
        "resume_text": resume_text[:12000],
        "word_count": len(resume_text.split()),
    }
