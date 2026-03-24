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
    return '\n'.join(lines).strip(), stderr.read().decode().strip(), exit_code

# Check if certbot is already installed
stdin, stdout, stderr = ssh.exec_command("certbot --version 2>&1")
stdout.channel.recv_exit_status()
ver = stdout.read().decode().strip()
print(f"certbot version: {ver}")

# Check current nginx config
out, err, rc = sudo_run("grep server_name /etc/nginx/sites-available/nafld")
print(f"nginx server_name: {out}")

# Check if SSL cert exists
out, err, rc = sudo_run("ls /etc/letsencrypt/live/ 2>&1")
print(f"certs: {out}")

ssh.close()
