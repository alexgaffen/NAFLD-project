import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

def sudo_run(cmd, password='123456'):
    full_cmd = f"echo '{password}' | sudo -S bash -c '{cmd}'"
    stdin, stdout, stderr = ssh.exec_command(full_cmd, get_pty=True)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    lines = [l for l in out.split('\n') if not l.strip().startswith('[sudo]')]
    return '\n'.join(lines).strip(), exit_code

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    return stdout.read().decode().strip(), exit_code

# Step 1: Update nginx server_name to use fibrosisai.org instead of aifibrosis.ca
print("=== Updating nginx server_name ===")
out, rc = sudo_run("sed -i 's/server_name .*/server_name 217.77.7.228 fibrosisai.org www.fibrosisai.org;/' /etc/nginx/sites-available/nafld")
print(f"  sed RC={rc}")

out, rc = sudo_run('nginx -t 2>&1')
print(f"  nginx -t: {out}")

out, rc = sudo_run('systemctl reload nginx')
print(f"  reload RC={rc}")

# Step 2: Run certbot for fibrosisai.org only (www not yet in DNS)
print("\n=== Running certbot ===")
out, rc = sudo_run('certbot --nginx -d fibrosisai.org --non-interactive --agree-tos -m alexgaffen@gmail.com --redirect 2>&1')
print(f"  certbot RC={rc}")
for line in out.split('\n'):
    line = line.strip()
    if line and not line.startswith('[sudo]'):
        print(f"  {line}")

# Step 3: Show final nginx config
print("\n=== Final nginx config ===")
out, rc = sudo_run('cat /etc/nginx/sites-available/nafld')
print(out)

ssh.close()
print("\nDone!")
