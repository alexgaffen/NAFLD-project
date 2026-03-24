import urllib.request, json

data = json.dumps({"username": "alexg", "password": "nafld2026"}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:5000/login",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)
resp = urllib.request.urlopen(req)
print(resp.read().decode())
