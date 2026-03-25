import { useState, useCallback } from "react";
import ImageSubmission from "./ImageSubmission";
import Login from "./Login";

function App() {
  const [user, setUser] = useState(() => sessionStorage.getItem("username"));

  const handleLogin = useCallback((username) => setUser(username), []);

  const handleLogout = useCallback(() => {
    sessionStorage.removeItem("access_token");
    sessionStorage.removeItem("refresh_token");
    sessionStorage.removeItem("username");
    setUser(null);
  }, []);

  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="app-root">
      {/* ── Top bar ── */}
      <header className="top-bar">
        <div className="top-bar-left">
          <svg className="brand-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="8" r="5"/><path d="M7 13 4.5 21h15L17 13"/><line x1="12" y1="3" x2="12" y2="0.5"/></svg>
          <span className="brand-name">fibrosisai</span>
        </div>
        <span className="top-bar-subtitle">
          AI-BASED UNSUPERVISED CLASSIFICATION AND QUANTIFICATION OF MOUSE LIVER FIBROSIS IN MASH
        </span>
        <span className="top-bar-pipeline">
          VGG16 FEATURE EXTRACTION → FUZZY C-MEANS CLUSTERING
        </span>
        <div className="user-badge">
          <span><svg className="user-icon-svg" viewBox="0 0 20 20" fill="currentColor"><path d="M10 10a4 4 0 100-8 4 4 0 000 8zm-7 8a7 7 0 0114 0H3z"/></svg> {user}</span>
          <button className="logout-btn" onClick={handleLogout}>Sign Out</button>
        </div>
      </header>

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
