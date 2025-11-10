import socket
from datetime import datetime, timezone
import time
client = socket.socket()
client.connect(("localhost", 9999))

client.send(b"johndoe")
ack = client.recv(1024)
client.send(b"john@123")
auth_response = client.recv(1024)

print(auth_response)
if auth_response != b"AUTH_OK":
    print(" Login failed.")
    client.close()
else:
    print(" Login successful.")

while True:
    msg = input(" Enter message (or 'exit'): ")

    if msg == "exit":
        break
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{msg}||{timestamp}"
    client.send(full_msg.encode())
   
    server_response = client.recv(1024).decode()
    print(" Server:", server_response)
    time.sleep(2)
    print("\n[Replaying same message...]")
    client.send(full_msg.encode())
    print(" Server:", client.recv(1024).decode())
