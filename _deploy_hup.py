"""Deploy on server: git pull, npm build, SIGHUP gunicorn (no sudo needed)."""
import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('217.77.7.228', username='alex', password='123456')

def run(cmd, timeout=600):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode()
    err = e.read().decode()
    return out.strip(), err.strip(), o.channel.recv_exit_status()

print("=== git pull ===")
out, err, rc = run("cd ~/NAFLD-project && git pull")
print(f"RC={rc}\n{out}\n{err}\n")

print("=== npm run build ===")
out, err, rc = run("cd ~/NAFLD-project/NAFLD/javascript/nafld-app && npm run build 2>&1 | tail -10")
print(f"RC={rc}\n{out}\n")

print("=== find & SIGHUP gunicorn master ===")
out, err, rc = run("MASTER=$(ps -eo pid,ppid,cmd | awk '/gunicorn.*main:app/ && !/awk/ && $2==1 {print $1; exit}'); "
                   "echo MASTER=$MASTER; "
                   "if [ -n \"$MASTER\" ]; then kill -HUP $MASTER && echo 'HUP sent'; fi")
print(out)
print(err)

time.sleep(4)
print("\n=== verify new workers ===")
out, _, _ = run("ps -eo pid,etime,cmd | grep gunicorn | grep -v grep")
print(out)

ssh.close()
print("\nDone.")
