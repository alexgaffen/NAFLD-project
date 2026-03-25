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

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    return stdout.read().decode().strip(), exit_code

# Step 1: Update React .env to use HTTPS domain
print("=== Updating React .env ===")
env_path = '/home/alex/NAFLD-project/NAFLD/javascript/nafld-app/.env'
with sftp.open(env_path, 'w') as f:
    f.write('REACT_APP_API_URL=https://fibrosisai.org\n')
out, rc = run(f'cat {env_path}')
print(f"  .env: {out}")

# Step 2: Rebuild React
print("\n=== Rebuilding React ===")
out, rc = run('cd ~/NAFLD-project/NAFLD/javascript/nafld-app && npm run build 2>&1 | tail -3')
print(f"  build RC={rc}")
print(f"  {out}")

# Step 3: Restart gunicorn
print("\n=== Restarting gunicorn ===")
out, rc = sudo_run('systemctl restart nafld')
print(f"  RC={rc}")

# Step 4: Reload nginx
out, rc = sudo_run('systemctl reload nginx')
print(f"  nginx reload RC={rc}")

# Step 5: Test HTTPS
print("\n=== Testing ===")
out, rc = sudo_run('curl -sI https://fibrosisai.org 2>&1 | head -3')
print(f"  HTTPS: {out}")

out, rc = sudo_run('curl -sI http://fibrosisai.org 2>&1 | head -3')
print(f"  HTTP redirect: {out}")

sftp.close()
ssh.close()
print("\nDone!")
