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

# Check nginx config
out, rc = sudo_run('cat /etc/nginx/sites-available/nafld')
print(f"=== nginx config ===")
print(out)

# Check certs
out, rc = sudo_run('ls -la /etc/letsencrypt/live/ 2>&1 || echo NO_CERTS')
print(f"\n=== certs ===")
print(out)

# Try certbot
print(f"\n=== Running certbot ===")
out, rc = sudo_run('certbot --nginx -d aifibrosis.ca -d www.aifibrosis.ca --non-interactive --agree-tos -m alexgaffen@gmail.com --redirect 2>&1')
print(f"RC={rc}")
print(out)

ssh.close()
