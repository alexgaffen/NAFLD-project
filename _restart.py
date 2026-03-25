import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

def sudo_run(cmd, password='123456'):
    stdin, stdout, stderr = ssh.exec_command(f"echo '{password}' | sudo -S {cmd}", get_pty=True)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    lines = [l for l in out.split('\n') if not l.strip().startswith('[sudo]')]
    return '\n'.join(lines).strip(), exit_code

out, rc = sudo_run('systemctl restart nafld 2>&1')
print(f"nafld restart: RC={rc}, {out}")
out, rc = sudo_run('systemctl status nafld --no-pager 2>&1 | head -5')
print(f"nafld status: {out}")
out, rc = sudo_run('systemctl reload nginx 2>&1')
print(f"nginx reload: RC={rc}, {out}")
out, rc = sudo_run('systemctl status nginx --no-pager 2>&1 | head -5')
print(f"nginx status: {out}")

ssh.close()
