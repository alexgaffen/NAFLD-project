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

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    return stdout.read().decode().strip(), exit_code

print("=== git pull ===")
out, rc = run('cd ~/NAFLD-project && git pull')
print(f"  RC={rc}")

print("=== npm run build ===")
out, rc = run('cd ~/NAFLD-project/NAFLD/javascript/nafld-app && npm run build 2>&1 | tail -3')
print(f"  RC={rc}, {out}")

print("=== restart ===")
out, rc = sudo_run('systemctl restart nafld && systemctl reload nginx')
print(f"  RC={rc}")

print("Done!")
ssh.close()
