import socket
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from user_auth import authenticate_user, init_db

init_db()
class MessageReceiver:
    def __init__(self):
        self.last_timestamp = None
        self.replay_window = timedelta(seconds=30)
       
    def verify_timestamp(self, timestamp_str):
        try:
            msg_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            now = datetime.utcnow()
            if self.last_timestamp and msg_time <= self.last_timestamp:
                return False, " Replay attack detected: Old timestamp."
            if abs(now - msg_time) > self.replay_window:
                return False, " Timestamp too old or too far in future."
            self.last_timestamp = msg_time

            return True, " Timestamp is valid."
        except ValueError:
            return False, " Invalid timestamp format."

server = socket.socket()
server.bind(('localhost', 9999))
server.listen(1)
print(" Server listening on port 9999...")
conn, addr = server.accept()
print(f" Connected by {addr}")

# Step 1: Receive login details
username = conn.recv(1024).decode()

conn.send(b"SEND_PASSWORD")  # send acknowledgment
password = conn.recv(1024).decode()

if authenticate_user(username, password):
    conn.send(b"AUTH_OK")
    print(" Authenticated:", username)

    # Step 2: Replay Protection
    receiver = MessageReceiver()
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        try:
            msg, timestamp_str = data.split("||")
        except ValueError:
            conn.send(b"ERROR: Invalid message format")
            continue
        valid, response = receiver.verify_timestamp(timestamp_str)
        print(f" [{username}] {msg} @ {timestamp_str} > {response}")
        conn.send(response.encode())
else:
    conn.send(b"AUTH_FAILED")
    print(" Authentication failed.")
    conn.close()
