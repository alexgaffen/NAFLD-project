import { useState } from "react";

const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

const FEATURES = [
    { title: "Colour Deconvolution", desc: "Isolates collagen fibers from PSR-stained tissue via optical density separation" },
    { title: "Adaptive Thresholding", desc: "Automatically determines the optimal binary threshold to quantify fibrosis extent" },
    { title: "Interactive Refinement", desc: "Fine-tune the fibrosis mask with a magnifying glass, per-area threshold adjustments, and live mini-map" },
    { title: "VGG16 Deep Features", desc: "Extracts high-level architectural patterns from tissue patches using a pretrained CNN" },
    { title: "Fuzzy C-Means Staging", desc: "Classifies the refined mask into probabilistic stages with worst-patch analysis" },
];

const Login = ({ onLogin }) => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        if (!username.trim() || !password) {
            setError("Username and password are required.");
            return;
        }

        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE}/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: username.trim(), password }),
            });
            const data = await res.json();
            if (!res.ok) {
                setError(data.error || "Login failed.");
                return;
            }
            sessionStorage.setItem("access_token", data.access_token);
            sessionStorage.setItem("refresh_token", data.refresh_token);
            sessionStorage.setItem("username", data.username);
            onLogin(data.username);
        } catch (err) {
            setError("Unable to connect to the server.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="login-backdrop">
            {/* Network constellation background */}
            <svg className="login-constellation" viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid slice">
                <defs>
                    <radialGradient id="node-glow"><stop offset="0%" stopColor="#4ecdc4" stopOpacity="0.6"/><stop offset="100%" stopColor="#4ecdc4" stopOpacity="0"/></radialGradient>
                </defs>
                {/* Edges */}
                <g stroke="#4ecdc4" strokeWidth="0.4" opacity="0.12">
                    <line x1="120" y1="80" x2="250" y2="140"/><line x1="250" y1="140" x2="400" y2="90"/>
                    <line x1="400" y1="90" x2="520" y2="180"/><line x1="520" y1="180" x2="680" y2="120"/>
                    <line x1="680" y1="120" x2="800" y2="200"/><line x1="800" y1="200" x2="900" y2="150"/>
                    <line x1="60" y1="350" x2="180" y2="280"/><line x1="180" y1="280" x2="320" y2="340"/>
                    <line x1="320" y1="340" x2="480" y2="300"/><line x1="480" y1="300" x2="600" y2="380"/>
                    <line x1="600" y1="380" x2="750" y2="320"/><line x1="750" y1="320" x2="880" y2="400"/>
                    <line x1="250" y1="140" x2="180" y2="280"/><line x1="520" y1="180" x2="480" y2="300"/>
                    <line x1="680" y1="120" x2="750" y2="320"/><line x1="400" y1="90" x2="320" y2="340"/>
                    <line x1="150" y1="480" x2="300" y2="520"/><line x1="300" y1="520" x2="500" y2="470"/>
                    <line x1="500" y1="470" x2="700" y2="530"/><line x1="700" y1="530" x2="850" y2="480"/>
                    <line x1="60" y1="350" x2="150" y2="480"/><line x1="600" y1="380" x2="500" y2="470"/>
                    <line x1="880" y1="400" x2="850" y2="480"/>
                </g>
                {/* Nodes */}
                <g fill="#4ecdc4">
                    {[[120,80],[250,140],[400,90],[520,180],[680,120],[800,200],[900,150],
                      [60,350],[180,280],[320,340],[480,300],[600,380],[750,320],[880,400],
                      [150,480],[300,520],[500,470],[700,530],[850,480]].map(([cx,cy],i) => (
                        <circle key={i} cx={cx} cy={cy} r={i % 3 === 0 ? 2.5 : 1.5} opacity={0.25 + (i % 4) * 0.08} />
                    ))}
                </g>
            </svg>

            <div className="login-split">
                {/* Left panel -- feature showcase */}
                <div className="login-hero">
                    <div className="login-hero-content">
                        <h1 className="login-brand">fibrosisai</h1>
                        <p className="login-tagline">AI-powered unsupervised classification and quantification of liver fibrosis</p>

                        <div className="login-features">
                            {FEATURES.map((f, i) => (
                                <div key={i} className="login-feature">
                                    <span className="login-feature-num">{String(i + 1).padStart(2, '0')}</span>
                                    <div>
                                        <div className="login-feature-title">{f.title}</div>
                                        <div className="login-feature-desc">{f.desc}</div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="login-pipeline-strip">
                            <span>PSR Staining</span>
                            <span className="login-pipe-arrow">&rarr;</span>
                            <span>Deconvolution</span>
                            <span className="login-pipe-arrow">&rarr;</span>
                            <span>Refinement</span>
                            <span className="login-pipe-arrow">&rarr;</span>
                            <span>VGG16</span>
                            <span className="login-pipe-arrow">&rarr;</span>
                            <span>FCM Staging</span>
                        </div>
                    </div>
                </div>

                {/* Right panel -- login form */}
                <div className="login-form-panel">
                    <form className="login-card" onSubmit={handleSubmit}>
                        <div className="login-card-glow" />

                        {/* Logo + title row */}
                        <div className="login-header-row">
                            <div className="login-icon">
                                <svg viewBox="0 0 200 160" className="liver-svg">
                                    <defs>
                                        <linearGradient id="lobe-left" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" stopColor="#5ee8df"/><stop offset="100%" stopColor="#1a8a82"/>
                                        </linearGradient>
                                        <linearGradient id="lobe-right" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" stopColor="#f08c6a"/><stop offset="100%" stopColor="#c0392b"/>
                                        </linearGradient>
                                        <linearGradient id="outline-left" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" stopColor="#7af5ed"/><stop offset="100%" stopColor="#3bb8b0"/>
                                        </linearGradient>
                                        <linearGradient id="outline-right" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" stopColor="#ff9e80"/><stop offset="100%" stopColor="#e05a3a"/>
                                        </linearGradient>
                                        <filter id="glow-cyan" x="-40%" y="-40%" width="180%" height="180%">
                                            <feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                                        </filter>
                                        <filter id="glow-red" x="-40%" y="-40%" width="180%" height="180%">
                                            <feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                                        </filter>
                                        <clipPath id="clip-left"><path d="M100,18 L100,140 L0,140 L0,0 L200,0 Z"/></clipPath>
                                        <clipPath id="clip-right"><path d="M100,18 L100,140 L200,140 L200,0 Z"/></clipPath>
                                    </defs>
                                    {/* Anatomical liver shape - large left lobe, smaller right lobe, with inferior notch */}
                                    <path id="liverShape" d="
                                        M100,18
                                        C82,12 55,14 35,24
                                        C18,34 8,52 10,72
                                        C12,88 22,104 38,116
                                        C52,126 70,130 85,128
                                        C92,126 96,118 100,110
                                        C104,118 108,126 115,128
                                        C130,130 148,122 158,112
                                        C168,100 174,84 170,68
                                        C166,52 154,36 138,26
                                        C122,14 108,14 100,18Z
                                    " fill="none"/>
                                    {/* Connective tissue / gallbladder notch detail */}
                                    <path d="M100,110 C96,120 92,132 88,140 C94,138 100,128 100,110Z" fill="#1a3a50" opacity="0.4"/>
                                    {/* Left lobe fill + outline */}
                                    <g clipPath="url(#clip-left)">
                                        <use href="#liverShape" fill="url(#lobe-left)" opacity="0.15"/>
                                        <use href="#liverShape" stroke="url(#outline-left)" strokeWidth="2" fill="none" filter="url(#glow-cyan)" opacity="0.8"/>
                                        {/* Sparse network - larger nodes, fewer connections */}
                                        <g stroke="#4ecdc4" strokeWidth="0.7" opacity="0.5">
                                            <line x1="30" y1="55" x2="50" y2="40"/><line x1="50" y1="40" x2="72" y2="35"/>
                                            <line x1="72" y1="35" x2="90" y2="45"/><line x1="30" y1="55" x2="45" y2="72"/>
                                            <line x1="45" y1="72" x2="65" y2="60"/><line x1="65" y1="60" x2="90" y2="45"/>
                                            <line x1="50" y1="40" x2="65" y2="60"/><line x1="65" y1="60" x2="85" y2="70"/>
                                            <line x1="45" y1="72" x2="60" y2="88"/><line x1="60" y1="88" x2="85" y2="70"/>
                                            <line x1="85" y1="70" x2="95" y2="55"/><line x1="90" y1="45" x2="95" y2="55"/>
                                            <line x1="60" y1="88" x2="75" y2="100"/><line x1="75" y1="100" x2="90" y2="90"/>
                                            <line x1="85" y1="70" x2="90" y2="90"/><line x1="30" y1="55" x2="25" y2="75"/>
                                            <line x1="25" y1="75" x2="45" y2="72"/><line x1="25" y1="75" x2="40" y2="95"/>
                                            <line x1="40" y1="95" x2="60" y2="88"/><line x1="40" y1="95" x2="55" y2="110"/>
                                            <line x1="55" y1="110" x2="75" y2="100"/><line x1="72" y1="35" x2="60" y2="28"/>
                                            <line x1="60" y1="28" x2="45" y2="32"/><line x1="45" y1="32" x2="50" y2="40"/>
                                            <line x1="45" y1="32" x2="30" y2="55"/>
                                        </g>
                                        <g fill="#4ecdc4">
                                            <circle cx="30" cy="55" r="2.5" opacity="0.7"/><circle cx="50" cy="40" r="2" opacity="0.6"/>
                                            <circle cx="72" cy="35" r="2.5" opacity="0.7"/><circle cx="90" cy="45" r="2" opacity="0.6"/>
                                            <circle cx="45" cy="72" r="2.5" opacity="0.7"/><circle cx="65" cy="60" r="3" opacity="0.8"/>
                                            <circle cx="85" cy="70" r="2" opacity="0.6"/><circle cx="60" cy="88" r="2.5" opacity="0.7"/>
                                            <circle cx="75" cy="100" r="2" opacity="0.6"/><circle cx="95" cy="55" r="2" opacity="0.6"/>
                                            <circle cx="25" cy="75" r="2" opacity="0.5"/><circle cx="40" cy="95" r="2" opacity="0.5"/>
                                            <circle cx="55" cy="110" r="2" opacity="0.5"/><circle cx="90" cy="90" r="1.8" opacity="0.5"/>
                                            <circle cx="60" cy="28" r="1.8" opacity="0.5"/><circle cx="45" cy="32" r="1.8" opacity="0.5"/>
                                        </g>
                                    </g>
                                    {/* Right lobe fill + outline */}
                                    <g clipPath="url(#clip-right)">
                                        <use href="#liverShape" fill="url(#lobe-right)" opacity="0.15"/>
                                        <use href="#liverShape" stroke="url(#outline-right)" strokeWidth="2" fill="none" filter="url(#glow-red)" opacity="0.8"/>
                                        {/* Dense mesh network */}
                                        <g stroke="#e8735a" strokeWidth="0.5" opacity="0.45">
                                            <line x1="105" y1="30" x2="118" y2="28"/><line x1="118" y1="28" x2="130" y2="32"/>
                                            <line x1="130" y1="32" x2="142" y2="30"/><line x1="142" y1="30" x2="155" y2="36"/>
                                            <line x1="105" y1="30" x2="110" y2="42"/><line x1="110" y1="42" x2="122" y2="38"/>
                                            <line x1="122" y1="38" x2="135" y2="44"/><line x1="135" y1="44" x2="148" y2="40"/>
                                            <line x1="148" y1="40" x2="160" y2="48"/><line x1="118" y1="28" x2="122" y2="38"/>
                                            <line x1="130" y1="32" x2="135" y2="44"/><line x1="142" y1="30" x2="148" y2="40"/>
                                            <line x1="155" y1="36" x2="160" y2="48"/><line x1="110" y1="42" x2="115" y2="55"/>
                                            <line x1="115" y1="55" x2="128" y2="52"/><line x1="128" y1="52" x2="140" y2="56"/>
                                            <line x1="140" y1="56" x2="155" y2="54"/><line x1="155" y1="54" x2="165" y2="60"/>
                                            <line x1="122" y1="38" x2="128" y2="52"/><line x1="135" y1="44" x2="140" y2="56"/>
                                            <line x1="148" y1="40" x2="155" y2="54"/><line x1="160" y1="48" x2="165" y2="60"/>
                                            <line x1="115" y1="55" x2="108" y2="68"/><line x1="108" y1="68" x2="120" y2="65"/>
                                            <line x1="120" y1="65" x2="132" y2="68"/><line x1="132" y1="68" x2="145" y2="66"/>
                                            <line x1="145" y1="66" x2="158" y2="70"/><line x1="128" y1="52" x2="120" y2="65"/>
                                            <line x1="140" y1="56" x2="132" y2="68"/><line x1="155" y1="54" x2="145" y2="66"/>
                                            <line x1="165" y1="60" x2="158" y2="70"/><line x1="108" y1="68" x2="112" y2="80"/>
                                            <line x1="112" y1="80" x2="125" y2="78"/><line x1="125" y1="78" x2="138" y2="82"/>
                                            <line x1="138" y1="82" x2="150" y2="78"/><line x1="120" y1="65" x2="125" y2="78"/>
                                            <line x1="132" y1="68" x2="138" y2="82"/><line x1="145" y1="66" x2="150" y2="78"/>
                                            <line x1="112" y1="80" x2="115" y2="92"/><line x1="115" y1="92" x2="130" y2="90"/>
                                            <line x1="130" y1="90" x2="142" y2="94"/><line x1="125" y1="78" x2="130" y2="90"/>
                                            <line x1="138" y1="82" x2="142" y2="94"/><line x1="115" y1="92" x2="118" y2="104"/>
                                            <line x1="118" y1="104" x2="130" y2="100"/><line x1="130" y1="90" x2="130" y2="100"/>
                                        </g>
                                        <g fill="#e8735a">
                                            <circle cx="105" cy="30" r="1.5" opacity="0.6"/><circle cx="118" cy="28" r="1.2" opacity="0.5"/>
                                            <circle cx="130" cy="32" r="1.5" opacity="0.6"/><circle cx="142" cy="30" r="1.2" opacity="0.5"/>
                                            <circle cx="155" cy="36" r="1.5" opacity="0.6"/><circle cx="110" cy="42" r="1.2" opacity="0.5"/>
                                            <circle cx="122" cy="38" r="1.5" opacity="0.6"/><circle cx="135" cy="44" r="1.8" opacity="0.7"/>
                                            <circle cx="148" cy="40" r="1.2" opacity="0.5"/><circle cx="160" cy="48" r="1.5" opacity="0.6"/>
                                            <circle cx="115" cy="55" r="1.5" opacity="0.6"/><circle cx="128" cy="52" r="1.8" opacity="0.7"/>
                                            <circle cx="140" cy="56" r="1.5" opacity="0.6"/><circle cx="155" cy="54" r="1.2" opacity="0.5"/>
                                            <circle cx="165" cy="60" r="1.2" opacity="0.5"/><circle cx="108" cy="68" r="1.5" opacity="0.6"/>
                                            <circle cx="120" cy="65" r="1.8" opacity="0.7"/><circle cx="132" cy="68" r="1.5" opacity="0.6"/>
                                            <circle cx="145" cy="66" r="1.2" opacity="0.5"/><circle cx="158" cy="70" r="1.5" opacity="0.6"/>
                                            <circle cx="112" cy="80" r="1.5" opacity="0.6"/><circle cx="125" cy="78" r="1.8" opacity="0.7"/>
                                            <circle cx="138" cy="82" r="1.5" opacity="0.6"/><circle cx="150" cy="78" r="1.2" opacity="0.5"/>
                                            <circle cx="115" cy="92" r="1.2" opacity="0.5"/><circle cx="130" cy="90" r="1.5" opacity="0.6"/>
                                            <circle cx="142" cy="94" r="1.2" opacity="0.5"/><circle cx="118" cy="104" r="1" opacity="0.4"/>
                                            <circle cx="130" cy="100" r="1" opacity="0.4"/>
                                        </g>
                                    </g>
                                    {/* Center transition glow */}
                                    <line x1="100" y1="18" x2="100" y2="110" stroke="#fff" strokeWidth="1" opacity="0.15"/>
                                </svg>
                            </div>
                            <h2 className="login-title">Sign In</h2>
                        </div>
                        <p className="login-subtitle">Access the fibrosis analysis platform</p>

                        <label className="login-label">
                            <span className="login-label-text">Username</span>
                            <div className="login-input-wrap">
                                <svg className="login-input-icon" viewBox="0 0 20 20" fill="currentColor"><path d="M10 10a4 4 0 100-8 4 4 0 000 8zm-7 8a7 7 0 0114 0H3z"/></svg>
                                <input
                                    className="login-input"
                                    type="text"
                                    placeholder="Enter your username"
                                    autoComplete="username"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    disabled={isLoading}
                                    autoFocus
                                />
                            </div>
                        </label>

                        <label className="login-label">
                            <span className="login-label-text">Password</span>
                            <div className="login-input-wrap">
                                <svg className="login-input-icon" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd"/></svg>
                                <input
                                    className="login-input"
                                    type="password"
                                    placeholder="Enter your password"
                                    autoComplete="current-password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    disabled={isLoading}
                                />
                            </div>
                        </label>

                        {error && <p className="login-error">{error}</p>}

                        <button className="login-btn" type="submit" disabled={isLoading}>
                            {isLoading ? (
                                <span className="login-btn-loading"><span className="login-spinner" /> Signing in…</span>
                            ) : "Sign In"}
                        </button>

                        <p className="login-footer">Whole-slide SVS/TIF support · 256×256 patch analysis · Interactive threshold tuning</p>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Login;
