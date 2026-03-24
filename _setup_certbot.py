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
    return '\n'.join(lines).strip(), stderr.read().decode().strip(), exit_code

# Check if certbot is installed
print("=== Checking certbot ===")
stdin, stdout, stderr = ssh.exec_command("which certbot 2>&1 || echo 'NOT FOUND'")
stdout.channel.recv_exit_status()
print(stdout.read().decode().strip())

# Try installing with snap instead (Ubuntu 24.04 prefers snap)
print("\n=== Installing certbot via snap ===")
out, err, rc = sudo_run("snap install --classic certbot 2>&1")
print(f"  RC={rc}")
print(f"  Output: {out[-800:]}")

# Link certbot
out, err, rc = sudo_run("ln -sf /snap/bin/certbot /usr/bin/certbot 2>&1")
print(f"  Link RC={rc}")

# Check certbot version
stdin, stdout, stderr = ssh.exec_command("certbot --version 2>&1")
stdout.channel.recv_exit_status()
print(f"\n  Version: {stdout.read().decode().strip()}")

# Now try getting the certificate
print("\n=== Obtaining SSL certificate ===")
out, err, rc = sudo_run("certbot --nginx -d aifibrosis.ca -d www.aifibrosis.ca --non-interactive --agree-tos -m alexgaffen@gmail.com --redirect 2>&1")
print(f"  certbot RC={rc}")
# Print full output for debugging
for line in out.split('\n'):
    line = line.strip()
    if line and not line.startswith('[sudo]'):
        print(f"  {line}")

ssh.close()
