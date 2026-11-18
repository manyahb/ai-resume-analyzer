import { useRef, useState } from "react";

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function Highlighted({ text, terms }) {
  if (!terms.length) return <>{text}</>;
  const re = new RegExp(`(?<![A-Za-z0-9])(${terms.map(esc).join("|")})(?![A-Za-z0-9])`, "gi");
  return text.split(re).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : part));
}

function Gauge({ value }) {
  const r = 70, c = 2 * Math.PI * r;
  return (
    <svg className="gauge" viewBox="0 0 180 180" role="img" aria-label={`ATS match score ${value} out of 100`}>
      <circle cx="90" cy="90" r={r} className="gauge-track" />
      <circle cx="90" cy="90" r={r} className="gauge-fill" strokeDasharray={c}
        strokeDashoffset={c * (1 - value / 100)} transform="rotate(-90 90 90)" />
      <text x="90" y="92" textAnchor="middle" className="gauge-num">{Math.round(value)}</text>
      <text x="90" y="116" textAnchor="middle" className="gauge-sub">out of 100</text>
    </svg>
  );
}

function buildReport(r, fileName) {
  const list = (a) => (a.length ? a.map((x) => `- ${x}`).join("\n") : "- none");
  return `# Resume Analysis Report

File: ${fileName}
ATS-style match score: ${r.ats_score}/100 (${r.verdict})
Raw TF-IDF cosine similarity: ${r.cosine_similarity}%

## Score breakdown
${Object.entries(r.breakdown).map(([k, v]) => `- ${k}: ${v}%`).join("\n")}

## Matched skills
${list(r.skills.matched.map((s) => s.name))}

## Skill gaps
${list(r.skills.missing.map((s) => s.name))}

## Keywords found
${list(r.keywords.matched)}

## Keywords missing
${list(r.keywords.missing)}

## Resume summary
${list(r.summary)}

## Suggestions
${list(r.suggestions)}
`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [jd, setJd] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [drag, setDrag] = useState(false);
  const input = useRef();

  const pick = (f) => {
    if (!f) return;
    if (!/\.(pdf|txt|md)$/i.test(f.name)) return setError("Upload a PDF or TXT resume.");
    setError("");
    setFile(f);
  };

  const analyze = async () => {
    if (!file) return setError("Add your resume first.");
    if (jd.trim().length < 30) return setError("Paste the full job description.");
    setBusy(true); setError("");
    try {
      const body = new FormData();
      body.append("resume", file);
      body.append("job_description", jd);
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Analysis failed.");
      setResult(data);
      setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (e) {
      setError(e.message === "Failed to fetch"
        ? "Can't reach the analysis server. Start the backend on port 8000." : e.message);
    } finally { setBusy(false); }
  };

  const download = () => {
    const blob = new Blob([buildReport(result, file.name)], { type: "text/markdown" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob), download: "resume-analysis-report.md" });
    a.click(); URL.revokeObjectURL(a.href);
  };

  return (
    <main>
      <header className="top">
        <h1>Resume Analyzer</h1>
        <p>See how your resume reads against a job description, and which words to add.</p>
      </header>

      <section className="inputs">
        <div
          className={`drop ${drag ? "over" : ""} ${file ? "has" : ""}`}
          onClick={() => input.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current.click()}
          tabIndex={0} role="button" aria-label="Upload resume"
        >
          <input ref={input} type="file" accept=".pdf,.txt,.md" hidden onChange={(e) => pick(e.target.files[0])} />
          {file ? (<><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(0)} KB. Click to replace.</span></>)
                : (<><strong>Drop your resume here</strong><span>PDF or TXT, up to 5 MB</span></>)}
        </div>
        <label className="jd">
          <span>Job description</span>
          <textarea value={jd} onChange={(e) => setJd(e.target.value)}
            placeholder="Paste the full job posting: responsibilities, requirements, tools." />
        </label>
      </section>

      {error && <p className="error" role="alert">{error}</p>}
      <button className="go" onClick={analyze} disabled={busy}>{busy ? "Analyzing…" : "Analyze resume"}</button>

      {result && (
        <section id="results" className="results">
          <div className="score-row">
            <Gauge value={result.ats_score} />
            <div>
              <h2>{result.verdict}</h2>
              <p className="muted">Raw TF-IDF cosine similarity: {result.cosine_similarity}%</p>
              <button className="ghost" onClick={download}>Download report</button>
            </div>
            <ul className="bars">
              {Object.entries(result.breakdown).map(([k, v]) => (
                <li key={k}>
                  <span>{k}</span><b>{v}%</b>
                  <i style={{ "--w": `${v}%` }} />
                </li>
              ))}
            </ul>
          </div>

          <div className="cols">
            <div>
              <h3>Skills you have that the job asks for</h3>
              <div className="chips">
                {result.skills.matched.length
                  ? result.skills.matched.map((s) => <span key={s.name} className="chip ok" title={s.category}>{s.name}</span>)
                  : <p className="muted">No overlap found with the skills list.</p>}
              </div>
            </div>
            <div>
              <h3>Skill gaps</h3>
              <div className="chips">
                {result.skills.missing.length
                  ? result.skills.missing.map((s) => <span key={s.name} className="chip miss" title={s.category}>{s.name}</span>)
                  : <p className="muted">No gaps found. Every skill in the posting is on your resume.</p>}
              </div>
            </div>
          </div>

          <h3>Resume summary</h3>
          <ul className="plain">{result.summary.length
            ? result.summary.map((s, i) => <li key={i}>{s}</li>)
            : <li className="muted">Not enough full sentences to summarize.</li>}</ul>

          <h3>What to change</h3>
          <ul className="plain">{result.suggestions.map((s, i) => <li key={i}>{s}</li>)}</ul>

          <h3>Your resume, with matching terms marked</h3>
          <pre className="doc"><Highlighted text={result.resume_text} terms={result.highlight_terms} /></pre>
          {result.keywords.missing.length > 0 && (
            <p className="muted">Not in your resume: {result.keywords.missing.join(", ")}</p>
          )}
        </section>
      )}
    </main>
  );
}
