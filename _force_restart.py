import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

cmds = [
    'find ~/NAFLD-project/NAFLD/src/py-src/__pycache__ -name "*.pyc" -delete 2>/dev/null; echo "pyc cleared"',
    'echo 123456 | sudo -S systemctl restart nafld.service',
    'sleep 2',
    'echo 123456 | sudo -S systemctl status nafld.service --no-pager -l 2>/dev/null | head -10',
]
for cmd in cmds:
    print(">>>", cmd[:60])
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    out = stdout.read().decode().strip()
    if out:
        print(out)

ssh.close()
print("Done.")
