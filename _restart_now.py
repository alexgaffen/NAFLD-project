"""Restart nafld service via interactive shell (handles sudo password reliably)."""
import paramiko
import time
import re

HOST = '217.77.7.228'
USER = 'alex'
PASS = '123456'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

chan = ssh.invoke_shell()
chan.settimeout(30)

# Wait for initial prompt
time.sleep(1.5)
while chan.recv_ready():
    chan.recv(65535)
    time.sleep(0.2)

MARK = '__DONE_TAG_9472__'

def run(cmd, timeout=60):
    full = f"{cmd}; echo {MARK}$?\n"
    chan.send(full)
    out = ''
    sent_pw = False
    end = time.time() + timeout
    while time.time() < end:
        if chan.recv_ready():
            chunk = chan.recv(65535).decode(errors='replace')
            out += chunk
            if not sent_pw and '[sudo] password' in out:
                chan.send(PASS + '\n')
                sent_pw = True
            if MARK in out:
                # also drain a tiny bit more
                time.sleep(0.2)
                while chan.recv_ready():
                    out += chan.recv(65535).decode(errors='replace')
                break
        else:
            time.sleep(0.1)
    m = re.search(re.escape(MARK) + r'(\d+)', out)
    rc = int(m.group(1)) if m else -1
    cleaned = re.sub(re.escape(MARK) + r'\d+', '', out)
    return cleaned.strip(), rc

print("=== restart nafld ===")
out, rc = run('sudo -S systemctl restart nafld', timeout=90)
print(f"RC={rc}\n{out}\n")

print("=== reload nginx ===")
out, rc = run('sudo -S systemctl reload nginx', timeout=30)
print(f"RC={rc}\n{out}\n")

print("=== nafld is-active ===")
out, rc = run('sudo -S systemctl is-active nafld', timeout=15)
print(f"RC={rc}\n{out}\n")

print("=== nafld status ===")
out, rc = run('sudo -S systemctl status nafld --no-pager -n 12', timeout=20)
print(out)

chan.close()
ssh.close()
