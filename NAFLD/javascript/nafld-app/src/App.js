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
          <svg className="brand-liver-svg" viewBox="0 0 200 160">
            <defs>
              <linearGradient id="tb-lobe-left" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#5ee8df"/><stop offset="100%" stopColor="#1a8a82"/></linearGradient>
              <linearGradient id="tb-lobe-right" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#f08c6a"/><stop offset="100%" stopColor="#c0392b"/></linearGradient>
              <linearGradient id="tb-ol" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#7af5ed"/><stop offset="100%" stopColor="#3bb8b0"/></linearGradient>
              <linearGradient id="tb-or" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#ff9e80"/><stop offset="100%" stopColor="#e05a3a"/></linearGradient>
              <filter id="tb-gc" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              <filter id="tb-gr" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              <clipPath id="tb-cl"><path d="M100,18 L100,140 L0,140 L0,0 L200,0 Z"/></clipPath>
              <clipPath id="tb-cr"><path d="M100,18 L100,140 L200,140 L200,0 Z"/></clipPath>
            </defs>
            <path id="tbLiver" d="M100,18 C82,12 55,14 35,24 C18,34 8,52 10,72 C12,88 22,104 38,116 C52,126 70,130 85,128 C92,126 96,118 100,110 C104,118 108,126 115,128 C130,130 148,122 158,112 C168,100 174,84 170,68 C166,52 154,36 138,26 C122,14 108,14 100,18Z" fill="none"/>
            <g clipPath="url(#tb-cl)">
              <use href="#tbLiver" fill="url(#tb-lobe-left)" opacity="0.15"/>
              <use href="#tbLiver" stroke="url(#tb-ol)" strokeWidth="2" fill="none" filter="url(#tb-gc)" opacity="0.8"/>
              <g stroke="#4ecdc4" strokeWidth="0.7" opacity="0.5">
                <line x1="30" y1="55" x2="50" y2="40"/><line x1="50" y1="40" x2="72" y2="35"/><line x1="72" y1="35" x2="90" y2="45"/><line x1="30" y1="55" x2="45" y2="72"/><line x1="45" y1="72" x2="65" y2="60"/><line x1="65" y1="60" x2="90" y2="45"/><line x1="50" y1="40" x2="65" y2="60"/><line x1="65" y1="60" x2="85" y2="70"/><line x1="45" y1="72" x2="60" y2="88"/><line x1="60" y1="88" x2="85" y2="70"/><line x1="85" y1="70" x2="95" y2="55"/><line x1="60" y1="88" x2="75" y2="100"/><line x1="75" y1="100" x2="90" y2="90"/><line x1="85" y1="70" x2="90" y2="90"/>
              </g>
              <g fill="#4ecdc4">
                <circle cx="30" cy="55" r="2.5" opacity="0.7"/><circle cx="50" cy="40" r="2" opacity="0.6"/><circle cx="72" cy="35" r="2.5" opacity="0.7"/><circle cx="90" cy="45" r="2" opacity="0.6"/><circle cx="45" cy="72" r="2.5" opacity="0.7"/><circle cx="65" cy="60" r="3" opacity="0.8"/><circle cx="85" cy="70" r="2" opacity="0.6"/><circle cx="60" cy="88" r="2.5" opacity="0.7"/><circle cx="75" cy="100" r="2" opacity="0.6"/><circle cx="90" cy="90" r="1.8" opacity="0.5"/>
              </g>
            </g>
            <g clipPath="url(#tb-cr)">
              <use href="#tbLiver" fill="url(#tb-lobe-right)" opacity="0.15"/>
              <use href="#tbLiver" stroke="url(#tb-or)" strokeWidth="2" fill="none" filter="url(#tb-gr)" opacity="0.8"/>
              <g stroke="#e8735a" strokeWidth="0.5" opacity="0.45">
                <line x1="105" y1="30" x2="118" y2="28"/><line x1="118" y1="28" x2="130" y2="32"/><line x1="130" y1="32" x2="142" y2="30"/><line x1="142" y1="30" x2="155" y2="36"/><line x1="105" y1="30" x2="110" y2="42"/><line x1="110" y1="42" x2="122" y2="38"/><line x1="122" y1="38" x2="135" y2="44"/><line x1="135" y1="44" x2="148" y2="40"/><line x1="148" y1="40" x2="160" y2="48"/><line x1="115" y1="55" x2="128" y2="52"/><line x1="128" y1="52" x2="140" y2="56"/><line x1="140" y1="56" x2="155" y2="54"/><line x1="155" y1="54" x2="165" y2="60"/><line x1="108" y1="68" x2="120" y2="65"/><line x1="120" y1="65" x2="132" y2="68"/><line x1="132" y1="68" x2="145" y2="66"/><line x1="112" y1="80" x2="125" y2="78"/><line x1="125" y1="78" x2="138" y2="82"/><line x1="115" y1="92" x2="130" y2="90"/>
              </g>
              <g fill="#e8735a">
                <circle cx="105" cy="30" r="1.5" opacity="0.6"/><circle cx="118" cy="28" r="1.2" opacity="0.5"/><circle cx="130" cy="32" r="1.5" opacity="0.6"/><circle cx="142" cy="30" r="1.2" opacity="0.5"/><circle cx="155" cy="36" r="1.5" opacity="0.6"/><circle cx="110" cy="42" r="1.2" opacity="0.5"/><circle cx="122" cy="38" r="1.5" opacity="0.6"/><circle cx="135" cy="44" r="1.8" opacity="0.7"/><circle cx="148" cy="40" r="1.2" opacity="0.5"/><circle cx="160" cy="48" r="1.5" opacity="0.6"/><circle cx="115" cy="55" r="1.5" opacity="0.6"/><circle cx="128" cy="52" r="1.8" opacity="0.7"/><circle cx="140" cy="56" r="1.5" opacity="0.6"/><circle cx="155" cy="54" r="1.2" opacity="0.5"/><circle cx="165" cy="60" r="1.2" opacity="0.5"/><circle cx="108" cy="68" r="1.5" opacity="0.6"/><circle cx="120" cy="65" r="1.8" opacity="0.7"/><circle cx="132" cy="68" r="1.5" opacity="0.6"/><circle cx="112" cy="80" r="1.5" opacity="0.6"/><circle cx="125" cy="78" r="1.8" opacity="0.7"/><circle cx="138" cy="82" r="1.5" opacity="0.6"/><circle cx="115" cy="92" r="1.2" opacity="0.5"/><circle cx="130" cy="90" r="1.5" opacity="0.6"/>
              </g>
            </g>
            <line x1="100" y1="18" x2="100" y2="110" stroke="#fff" strokeWidth="1" opacity="0.15"/>
          </svg>
          <span className="brand-name">fibrosisai</span>
        </div>
        <span className="top-bar-subtitle">
          AI-BASED UNSUPERVISED CLASSIFICATION AND QUANTIFICATION OF MOUSE LIVER FIBROSIS IN MASH
        </span>
        <span className="top-bar-pipeline">
          VGG16 FEATURE EXTRACTION → FCM CLUSTERING
        </span>
        <div className="user-badge">
          <svg className="user-icon-svg" viewBox="0 0 20 20" fill="currentColor"><path d="M10 10a4 4 0 100-8 4 4 0 000 8zm-7 8a7 7 0 0114 0H3z"/></svg>
          <span className="user-badge-name">{user}</span>
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
