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
          <span className="brand-icon">🔬</span>
          <span className="brand-name">AIFIBROSIS</span>
        </div>
        <span className="top-bar-subtitle">
          AI-BASED UNSUPERVISED CLASSIFICATION AND QUANTIFICATION OF MOUSE LIVER FIBROSIS IN MASH
        </span>
        <span className="top-bar-pipeline">
          VGG16 FEATURE EXTRACTION → FUZZY C-MEANS CLUSTERING
        </span>
        <div className="user-badge">
          <span>👤 {user}</span>
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
