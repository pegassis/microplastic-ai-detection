import { useCallback, useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL ?? "";

// Colour-blind-friendly palette (Okabe-Ito), one colour per polymer.
const COLORS = {
  ABS: "#E69F00",
  Nylon: "#56B4E9",
  PE: "#009E73",
  PET: "#CC79A7",
  PS: "#0072B2",
  PVC: "#D55E00",
};

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function toCsv(result) {
  const rows = [["class", "confidence", "x1", "y1", "x2", "y2", "width_px", "height_px"]];
  result.detections.forEach((d) => rows.push([d.class, d.confidence, ...d.bbox, d.width_px, d.height_px]));
  return rows.map((r) => r.join(",")).join("\n");
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [view, setView] = useState("annotated");
  const [history, setHistory] = useState({ enabled: false, items: [] });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const loadHistory = useCallback(() => {
    fetch(`${API}/api/history`).then((r) => r.json()).then(setHistory).catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`${API}/api/health`).then((r) => r.json()).then(setHealth).catch(() =>
      setHealth({ status: "unreachable", error: "Cannot reach the backend. Is uvicorn running on port 8000?" })
    );
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!file) return setPreviewUrl(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function pick(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) return setError("Please choose an image file (JPG, PNG, BMP or TIFF).");
    setError(null);
    setResult(null);
    setView("annotated");
    setFile(f);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("conf", String(conf));
      const res = await fetch(`${API}/api/detect`, { method: "POST", body });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.detail || `Request failed (${res.status})`);
      setResult(json);
      setView("annotated");
      loadHistory();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const modelReady = health?.model_loaded;
  const maxCount = result ? Math.max(1, ...Object.values(result.counts)) : 1;

  return (
    <div className="page">
      <header>
        <h1>Microplastic Detector</h1>
        <p className="sub">
          Upload a Nile&nbsp;Red fluorescence microscope image. YOLO11 locates particles and labels them as ABS, Nylon,
          PE, PET, PS or PVC.
        </p>
      </header>

      {health && !modelReady && (
        <div className="banner warn">
          {health.error || "Model not loaded."} Detection is disabled until a model is available.
        </div>
      )}

      <main className="grid">
        <section className="card">
          <h2>1. Image</h2>
          <div
            className={`drop ${dragging ? "over" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]); }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
          >
            {previewUrl ? (
              <img src={previewUrl} alt="Selected sample" />
            ) : (
              <span>Drop an image here or click to browse</span>
            )}
            <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files?.[0])} />
          </div>
          {file && <p className="fname">{file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB</p>}

          <label className="slider">
            <span>Confidence threshold: <b>{conf.toFixed(2)}</b></span>
            <input type="range" min="0.05" max="0.9" step="0.05" value={conf} onChange={(e) => setConf(+e.target.value)} />
            <small>Lower finds more particles but adds false positives.</small>
          </label>

          <button className="primary" disabled={!file || loading || !modelReady} onClick={analyze}>
            {loading ? "Analysing…" : "Detect microplastics"}
          </button>
          {error && <div className="banner err" role="alert">{error}</div>}
        </section>

        <section className="card">
          <h2>2. Results</h2>
          {!result && <p className="muted">Results appear here after analysis.</p>}
          {result && (
            <>
              <div className="tabs">
                <button className={view === "annotated" ? "on" : ""} onClick={() => setView("annotated")}>Detections</button>
                <button className={view === "original" ? "on" : ""} onClick={() => setView("original")}>Original</button>
              </div>
              <img className="resultimg" src={view === "annotated" ? result.annotated_image : previewUrl} alt={view} />

              <div className="stats">
                <div><b>{result.total}</b><span>particles</span></div>
                <div><b>{result.inference_ms}</b><span>ms</span></div>
                <div><b>{result.image_size.join("×")}</b><span>pixels</span></div>
              </div>

              <h3>Count by polymer</h3>
              <ul className="bars">
                {Object.entries(result.counts).map(([name, n]) => (
                  <li key={name}>
                    <span className="lbl"><i style={{ background: COLORS[name] || "#888" }} />{name}</span>
                    <span className="track"><span style={{ width: `${(n / maxCount) * 100}%`, background: COLORS[name] || "#888" }} /></span>
                    <span className="num">{n}{result.mean_confidence[name] != null && <small> · {Math.round(result.mean_confidence[name] * 100)}%</small>}</span>
                  </li>
                ))}
              </ul>
              <p className="muted small">Percentages are the mean model confidence per class.</p>

              <div className="row">
                <button onClick={() => download("detections.csv", toCsv(result), "text/csv")}>Download CSV</button>
                <button onClick={() => {
                  const { annotated_image, ...rest } = result;
                  download("analysis.json", JSON.stringify(rest, null, 2), "application/json");
                }}>Download JSON</button>
              </div>
            </>
          )}
        </section>
      </main>

      {history.enabled && history.items.length > 0 && (
        <section className="card">
          <h2>Recent analyses</h2>
          <table>
            <thead><tr><th>When</th><th>File</th><th>Particles</th><th>Threshold</th></tr></thead>
            <tbody>
              {history.items.map((h) => (
                <tr key={h.id}>
                  <td>{new Date(h.created_at).toLocaleString()}</td>
                  <td>{h.filename}</td>
                  <td>{h.total}</td>
                  <td>{h.conf_threshold}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <footer>
        Research prototype. The model only knows the six polymers and the Nile Red fluorescence imaging it was trained on,
        and appearance alone cannot confirm polymer identity. Verify important results with FTIR or Raman spectroscopy.
      </footer>
    </div>
  );
}
