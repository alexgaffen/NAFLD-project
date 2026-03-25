import { useState } from "react";

const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

const FEATURES = [
    { title: "Colour Deconvolution", desc: "Isolates collagen fibers from PSR-stained tissue via optical density separation" },
    { title: "Adaptive Thresholding", desc: "Automatically determines the optimal binary threshold to quantify fibrosis extent" },
    { title: "VGG16 Deep Features", desc: "Extracts high-level architectural patterns from tissue patches using a pretrained CNN" },
    { title: "Fuzzy C-Means Staging", desc: "Classifies fibrosis into probabilistic stages \u2014 no hard boundaries, just a continuous spectrum" },
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

                        {/* Custom liver icon */}
                        <div className="login-icon">
                            <svg viewBox="0 0 120 100" className="liver-svg">
                                <defs>
                                    <linearGradient id="lobe-left" x1="0" y1="0" x2="1" y2="1">
                                        <stop offset="0%" stopColor="#4ecdc4"/><stop offset="100%" stopColor="#2a9d8f"/>
                                    </linearGradient>
                                    <linearGradient id="lobe-right" x1="0" y1="0" x2="1" y2="1">
                                        <stop offset="0%" stopColor="#e76f51"/><stop offset="100%" stopColor="#c0392b"/>
                                    </linearGradient>
                                    <linearGradient id="center-glow" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#fff" stopOpacity="0.9"/><stop offset="50%" stopColor="#4ecdc4" stopOpacity="0.5"/><stop offset="100%" stopColor="#fff" stopOpacity="0.9"/>
                                    </linearGradient>
                                    <clipPath id="clip-left"><rect x="0" y="0" width="60" height="100"/></clipPath>
                                    <clipPath id="clip-right"><rect x="60" y="0" width="60" height="100"/></clipPath>
                                </defs>
                                {/* Liver shape path */}
                                <path id="liverPath" d="M60,14 C42,10 22,16 14,32 C8,44 10,58 18,68 C26,78 40,82 52,78 C56,76 58,72 60,68 C62,72 64,76 68,78 C80,82 94,78 102,68 C110,58 112,44 106,32 C98,16 78,10 60,14Z" fill="none"/>
                                {/* Left lobe: cyan with sparse network */}
                                <g clipPath="url(#clip-left)">
                                    <use href="#liverPath" fill="url(#lobe-left)" opacity="0.85"/>
                                    <g stroke="#b2f0ea" strokeWidth="0.6" opacity="0.45">
                                        <line x1="22" y1="35" x2="38" y2="28"/>
                                        <line x1="38" y1="28" x2="50" y2="40"/>
                                        <line x1="22" y1="35" x2="30" y2="52"/>
                                        <line x1="30" y1="52" x2="50" y2="40"/>
                                        <line x1="30" y1="52" x2="45" y2="62"/>
                                        <line x1="50" y1="40" x2="55" y2="55"/>
                                    </g>
                                    <g fill="#e0faf7" opacity="0.6">
                                        <circle cx="22" cy="35" r="1.8"/><circle cx="38" cy="28" r="1.3"/>
                                        <circle cx="50" cy="40" r="1.8"/><circle cx="30" cy="52" r="1.3"/>
                                        <circle cx="45" cy="62" r="1.5"/><circle cx="55" cy="55" r="1.0"/>
                                    </g>
                                </g>
                                {/* Right lobe: red-orange with dense mesh */}
                                <g clipPath="url(#clip-right)">
                                    <use href="#liverPath" fill="url(#lobe-right)" opacity="0.85"/>
                                    <g stroke="#f4a896" strokeWidth="0.4" opacity="0.4">
                                        <line x1="65" y1="30" x2="75" y2="25"/><line x1="75" y1="25" x2="85" y2="30"/>
                                        <line x1="85" y1="30" x2="95" y2="28"/><line x1="65" y1="30" x2="70" y2="40"/>
                                        <line x1="70" y1="40" x2="80" y2="38"/><line x1="80" y1="38" x2="90" y2="42"/>
                                        <line x1="90" y1="42" x2="98" y2="38"/><line x1="75" y1="25" x2="80" y2="38"/>
                                        <line x1="85" y1="30" x2="90" y2="42"/><line x1="70" y1="40" x2="75" y2="50"/>
                                        <line x1="75" y1="50" x2="85" y2="48"/><line x1="85" y1="48" x2="95" y2="52"/>
                                        <line x1="80" y1="38" x2="85" y2="48"/><line x1="90" y1="42" x2="95" y2="52"/>
                                        <line x1="75" y1="50" x2="80" y2="60"/><line x1="80" y1="60" x2="88" y2="58"/>
                                        <line x1="85" y1="48" x2="88" y2="58"/><line x1="80" y1="60" x2="75" y2="68"/>
                                        <line x1="65" y1="55" x2="75" y2="50"/><line x1="65" y1="55" x2="70" y2="65"/>
                                        <line x1="70" y1="65" x2="75" y2="68"/>
                                    </g>
                                    <g fill="#fcd5cc" opacity="0.5">
                                        <circle cx="65" cy="30" r="1"/><circle cx="75" cy="25" r="1.2"/>
                                        <circle cx="85" cy="30" r="1"/><circle cx="95" cy="28" r="0.8"/>
                                        <circle cx="70" cy="40" r="1.2"/><circle cx="80" cy="38" r="1"/>
                                        <circle cx="90" cy="42" r="1"/><circle cx="98" cy="38" r="0.8"/>
                                        <circle cx="75" cy="50" r="1.2"/><circle cx="85" cy="48" r="1"/>
                                        <circle cx="95" cy="52" r="0.8"/><circle cx="80" cy="60" r="1"/>
                                        <circle cx="88" cy="58" r="0.8"/><circle cx="65" cy="55" r="1"/>
                                        <circle cx="70" cy="65" r="0.8"/><circle cx="75" cy="68" r="1"/>
                                    </g>
                                </g>
                                {/* Center radiant dividing line */}
                                <line x1="60" y1="14" x2="60" y2="68" stroke="url(#center-glow)" strokeWidth="1.8" opacity="0.7"/>
                            </svg>
                        </div>

                        <h2 className="login-title">Welcome back</h2>
                        <p className="login-subtitle">Sign in to access the analysis platform</p>

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
