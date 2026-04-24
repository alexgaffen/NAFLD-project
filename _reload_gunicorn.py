"""Reload gunicorn workers via SIGHUP (no sudo needed since user alex owns it)."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

def run(c):
    _, o, e = ssh.exec_command(c)
    return o.read().decode().strip(), e.read().decode().strip(), o.channel.recv_exit_status()

print("--- gunicorn processes ---")
out, err, rc = run("ps -eo pid,ppid,user,cmd --sort=pid | grep 'gunicorn.*main:app' | grep -v grep")
print(out)

# Master = the one with lowest pid AND ppid is not another gunicorn (typically ppid=1 under systemd)
print("\n--- Sending SIGHUP to gunicorn master (graceful reload, re-imports app) ---")
out, err, rc = run("pkill -HUP -f 'gunicorn.*main:app' -n -u alex 2>&1 || true; "
                   "MASTER=$(ps -eo pid,ppid,cmd | awk '/gunicorn.*main:app/ && !/awk/ && $2==1 {print $1; exit}'); "
                   "echo MASTER_PID=$MASTER; "
                   "if [ -n \"$MASTER\" ]; then kill -HUP $MASTER && echo 'HUP sent OK'; fi")
print(out)
print('stderr:', err, 'rc:', rc)

print("\n--- Wait a moment, then check status ---")
import time; time.sleep(3)
out, err, rc = run("systemctl status nafld --no-pager -n 6 | head -20")
print(out)

print("\n--- Confirm workers restarted (look for fresh 'Booting worker' log) ---")
out, err, rc = run("journalctl -u nafld --no-pager -n 20 --since '2 minutes ago' 2>/dev/null || journalctl -u nafld --no-pager -n 20")
print(out)

ssh.close()
