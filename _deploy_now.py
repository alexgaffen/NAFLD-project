import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

cmds = [
    'cd ~/NAFLD-project && git pull',
    'cd ~/NAFLD-project/NAFLD/javascript/nafld-app && npm run build 2>&1 | tail -5',
    'echo 123456 | sudo -S systemctl restart nafld.service',
]
for cmd in cmds:
    print(">>>", cmd[:60])
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=300)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err and 'password' not in err.lower():
        print(err)
    print()

stdin2, stdout2, _ = ssh.exec_command('echo 123456 | sudo -S systemctl status nafld.service --no-pager -l 2>/dev/null | head -5', timeout=15)
print(stdout2.read().decode().strip())
ssh.close()
print("Deploy complete.")
