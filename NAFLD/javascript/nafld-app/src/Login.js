import { useState } from "react";

const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

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
            // Store tokens
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
            <form className="login-card" onSubmit={handleSubmit}>
                <div className="login-icon">🔬</div>
                <h1 className="login-title">fibrosisai</h1>
                <p className="login-subtitle">Sign in to continue</p>

                <label className="login-label">
                    Username
                    <input
                        className="login-input"
                        type="text"
                        autoComplete="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        disabled={isLoading}
                        autoFocus
                    />
                </label>

                <label className="login-label">
                    Password
                    <input
                        className="login-input"
                        type="password"
                        autoComplete="current-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={isLoading}
                    />
                </label>

                {error && <p className="login-error">{error}</p>}

                <button className="login-btn" type="submit" disabled={isLoading}>
                    {isLoading ? "Signing in…" : "Sign In"}
                </button>
            </form>
        </div>
    );
};

export default Login;
