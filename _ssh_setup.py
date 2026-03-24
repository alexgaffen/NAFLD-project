import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')
pubkey = open(r'C:\Users\alexg\.ssh\id_rsa.pub').read().strip()
for cmd in ['mkdir -p ~/.ssh', 'chmod 700 ~/.ssh', 'touch ~/.ssh/authorized_keys', 'chmod 600 ~/.ssh/authorized_keys']:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.recv_exit_status()
stdin, stdout, stderr = ssh.exec_command('cat ~/.ssh/authorized_keys')
existing = stdout.read().decode()
if pubkey[:50] not in existing:
    stdin, stdout, stderr = ssh.exec_command(f'echo "{pubkey}" >> ~/.ssh/authorized_keys')
    stdout.channel.recv_exit_status()
    print('SSH key installed')
else:
    print('SSH key already present')
ssh.close()
