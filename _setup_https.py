import paramiko, sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

def sudo_run(cmd, password='123456'):
    """Run a command with sudo, feeding password via stdin."""
    full_cmd = f"echo '{password}' | sudo -S {cmd}"
    stdin, stdout, stderr = ssh.exec_command(full_cmd, get_pty=True)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    # Filter out the password prompt from output
    lines = [l for l in out.split('\n') if not l.strip().startswith('[sudo]')]
    return '\n'.join(lines).strip(), err.strip(), exit_code

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    return stdout.read().decode().strip(), stderr.read().decode().strip(), exit_code

# Step 1: Update nginx config — add domain names
print("=== Updating nginx server_name ===")
out, err, rc = sudo_run("sed -i 's/server_name 217.77.7.228;/server_name 217.77.7.228 aifibrosis.ca www.aifibrosis.ca;/' /etc/nginx/sites-available/nafld")
print(f"  RC={rc}")

# Verify
out, err, rc = sudo_run("nginx -t")
print(f"  nginx -t: {out}")

# Step 2: Install certbot
print("\n=== Installing certbot ===")
out, err, rc = sudo_run("apt-get update -qq && apt-get install -y -qq certbot python3-certbot-nginx")
print(f"  Install RC={rc}")

# Step 3: Obtain SSL certificate
print("\n=== Obtaining SSL certificate ===")
out, err, rc = sudo_run("certbot --nginx -d aifibrosis.ca -d www.aifibrosis.ca --non-interactive --agree-tos -m alexgaffen@gmail.com --redirect")
print(f"  certbot RC={rc}")
print(f"  Output: {out[-500:] if len(out) > 500 else out}")

# Step 4: Reload nginx
print("\n=== Reloading nginx ===")
out, err, rc = sudo_run("systemctl reload nginx")
print(f"  RC={rc}")

# Step 5: Check nginx config
out, err, rc = sudo_run("cat /etc/nginx/sites-available/nafld")
print(f"\n=== Final nginx config ===\n{out}")

ssh.close()
print("\nDone!")
