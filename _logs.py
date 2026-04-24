import paramiko
s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect('217.77.7.228', username='alex', password='123456')
_, o, e = s.exec_command("journalctl -u nafld --no-pager -n 80 --since '5 minutes ago'")
print(o.read().decode())
s.close()
