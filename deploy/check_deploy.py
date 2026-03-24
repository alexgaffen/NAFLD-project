import subprocess, re

# Check baked URLs in JS bundle
result = subprocess.run(
    ["grep", "-roh", r"http://[0-9.]*[:/0-9]*", "/home/alex/NAFLD-project/NAFLD/javascript/nafld-app/build/static/js/"],
    capture_output=True, text=True
)
urls = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
print("URLs in JS bundle:", urls)

# Check .env
with open("/home/alex/NAFLD-project/NAFLD/javascript/nafld-app/.env") as f:
    print(".env:", f.read().strip())

# Test login directly
import urllib.request, json
try:
    data = json.dumps({"username": "alexg", "password": "nafld2026"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:5000/login",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    resp = urllib.request.urlopen(req)
    body = json.loads(resp.read().decode())
    print("Login OK, got token:", body.get("access_token", "")[:20] + "...")
except Exception as e:
    print("Login FAILED:", e)

# Check users in DB
import sqlite3
conn = sqlite3.connect("/home/alex/NAFLD-project/NAFLD/src/py-src/users.db")
users = conn.execute("SELECT id, username FROM users").fetchall()
print("Users in DB:", users)
conn.close()
