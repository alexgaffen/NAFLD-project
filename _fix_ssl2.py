import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')
sftp = ssh.open_sftp()

def sudo_run(cmd, password='123456'):
    stdin, stdout, stderr = ssh.exec_command(f"echo '{password}' | sudo -S {cmd}", get_pty=True)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    lines = [l for l in out.split('\n') if not l.strip().startswith('[sudo]')]
    return '\n'.join(lines).strip(), exit_code

# Step 1: Write updated nginx config to a temp file, then sudo move it
nginx_conf = """server {
    listen 80;
    server_name 217.77.7.228 fibrosisai.org www.fibrosisai.org;
    client_max_body_size 2G;
    root /home/alex/NAFLD-project/NAFLD/javascript/nafld-app/build;
    index index.html;

    location ~ ^/(analyze|analyze-stream|preview|rethreshold|rethreshold-area|reset-area|undo-area|download-single|upload|largefile|login|refresh|me|home|fake|download|fullFileUpload) {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering off;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
"""

print("=== Writing nginx config ===")
with sftp.open('/tmp/nafld_nginx', 'w') as f:
    f.write(nginx_conf)
out, rc = sudo_run('cp /tmp/nafld_nginx /etc/nginx/sites-available/nafld')
print(f"  copy RC={rc}")

out, rc = sudo_run('nginx -t 2>&1')
print(f"  nginx -t: {out}")

out, rc = sudo_run('systemctl reload nginx')
print(f"  reload RC={rc}")

# Step 2: Install the SSL cert
print("\n=== Installing SSL certificate ===")
out, rc = sudo_run('certbot install --cert-name fibrosisai.org --nginx --redirect --non-interactive 2>&1')
print(f"  certbot install RC={rc}")
for line in out.split('\n'):
    line = line.strip()
    if line and not line.startswith('[sudo]'):
        print(f"  {line}")

# Step 3: Reload nginx again
out, rc = sudo_run('systemctl reload nginx')
print(f"\n  final reload RC={rc}")

# Step 4: Show final config
print("\n=== Final nginx config ===")
out, rc = sudo_run('cat /etc/nginx/sites-available/nafld')
print(out)

# Step 5: Test
print("\n=== Testing HTTPS ===")
out, rc = sudo_run('curl -sI https://fibrosisai.org 2>&1 | head -5')
print(out)

sftp.close()
ssh.close()
print("\nDone!")
