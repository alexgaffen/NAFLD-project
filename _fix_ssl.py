import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

def sudo_run(cmd, password='123456'):
    full_cmd = f"echo '{password}' | sudo -S bash -c \"{cmd}\""
    stdin, stdout, stderr = ssh.exec_command(full_cmd, get_pty=True)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    lines = [l for l in out.split('\n') if not l.strip().startswith('[sudo]')]
    return '\n'.join(lines).strip(), exit_code

# Step 1: Fix nginx server_name using python on the server
print("=== Fixing nginx server_name ===")
fix_cmd = """python3 -c "
p='/etc/nginx/sites-available/nafld'
t=open(p).read()
t=t.replace('server_name 217.77.7.228 aifibrosis.ca www.aifibrosis.ca;','server_name 217.77.7.228 fibrosisai.org www.fibrosisai.org;')
open(p,'w').write(t)
print('Updated')
" """
out, rc = sudo_run(fix_cmd)
print(f"  RC={rc}, {out}")

# Verify
out, rc = sudo_run("grep server_name /etc/nginx/sites-available/nafld")
print(f"  server_name: {out}")

# Test nginx
out, rc = sudo_run("nginx -t 2>&1")
print(f"  nginx -t: {out}")

# Step 2: Install the cert that certbot already obtained
print("\n=== Installing certificate ===")
out, rc = sudo_run("certbot install --cert-name fibrosisai.org --nginx --redirect --non-interactive 2>&1")
print(f"  certbot install RC={rc}")
for line in out.split('\n'):
    line = line.strip()
    if line and not line.startswith('[sudo]'):
        print(f"  {line}")

# Step 3: Show final config
print("\n=== Final nginx config ===")
out, rc = sudo_run("cat /etc/nginx/sites-available/nafld")
print(out)

# Step 4: Test HTTPS
print("\n=== Testing ===")
out, rc = sudo_run("curl -sI https://fibrosisai.org 2>&1 | head -5")
print(out)

ssh.close()
print("\nDone!")
