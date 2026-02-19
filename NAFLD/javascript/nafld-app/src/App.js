import ImageSubmission from "./ImageSubmission";

function App() {
  return (
    <div className="app-root">
      {/* ── Top bar ── */}
      <header className="top-bar">
        <div className="top-bar-left">
          <span className="brand-icon">🔬</span>
          <span className="brand-name">AIFIBROSIS</span>
        </div>
        <span className="top-bar-subtitle">
          AI-BASED UNSUPERVISED CLASSIFICATION AND QUANTIFICATION OF MOUSE LIVER FIBROSIS IN MASH
        </span>
      </header>

      <p className="pipeline-label">
        QUANTIFICATION PIPELINE: VGG16 FEATURE EXTRACTION → FUZZY C-MEANS CLUSTERING
      </p>

      {/* ── Main content ── */}
      <ImageSubmission />

      {/* ── Footer logos ── */}
      <footer className="logo-footer">
        <img src="/Images/McMaster.png" alt="McMaster" className="footer-logo" />
        <img src="/Images/ICELAB.png"   alt="ICE Lab"  className="footer-logo" />
        <img src="/Images/Heersink.png" alt="Heersink" className="footer-logo" />
      </footer>
    </div>
  );
}

export default App;
