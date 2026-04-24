import paramiko
s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect('217.77.7.228', username='alex', password='123456')
_, o, e = s.exec_command(
    "ls -la /etc/nginx/sites-available/nafld /etc/nginx/sites-enabled/nafld; "
    "echo '---'; "
    "id; "
    "echo '---try write test:'; "
    "test -w /etc/nginx/sites-available/nafld && echo WRITABLE || echo NOT_WRITABLE; "
    "echo '---sudo test:'; "
    "sudo -n -l 2>&1 | head -20"
)
print(o.read().decode())
print('---STDERR---')
print(e.read().decode())
s.close()
