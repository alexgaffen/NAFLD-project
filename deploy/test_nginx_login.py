import urllib.request, json

# Test login through NGINX (port 80)  
data = json.dumps({"username": "alexg", "password": "nafld2026"}).encode()
req = urllib.request.Request(
    "http://127.0.0.1/login",  # through nginx
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    resp = urllib.request.urlopen(req)
    body = json.loads(resp.read().decode())
    print("Login via NGINX OK:", list(body.keys()))
except urllib.error.HTTPError as e:
    print(f"Login via NGINX FAILED: {e.code} {e.read().decode()}")
