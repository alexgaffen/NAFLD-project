import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    return stdout.read().decode().strip(), stderr.read().decode().strip(), exit_code

out, err, rc = run('cd ~/NAFLD-project/NAFLD/src/py-src && ~/nafld-venv/bin/python auth.py add wanglab wanglab2026 2>&1')
print(f"RC={rc}")
print(out)
if err:
    print(f"STDERR: {err}")

ssh.close()
