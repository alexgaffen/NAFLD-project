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

# Step 1: Rebuild React frontend
print("=== Building React app ===")
out, rc = run('cd ~/NAFLD-project/NAFLD/javascript/nafld-app && npm run build 2>&1 | tail -5')
print(f"  RC={rc}")
print(f"  {out}")

# Step 2: Restart gunicorn
print("\n=== Restarting gunicorn ===")
out, rc = sudo_run('systemctl restart nafld')
print(f"  RC={rc}")

# Step 3: Reload nginx
print("\n=== Reloading nginx ===")
out, rc = sudo_run('systemctl reload nginx')
print(f"  RC={rc}")

# Step 4: Verify
out, rc = sudo_run('systemctl status nafld --no-pager -l 2>&1 | head -10')
print(f"\n=== nafld service ===")
print(out)

out, rc = sudo_run('systemctl status nginx --no-pager -l 2>&1 | head -5')
print(f"\n=== nginx ===")
print(out)

ssh.close()
print("\nDone!")
