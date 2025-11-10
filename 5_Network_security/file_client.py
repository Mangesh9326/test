import socket
import json
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AES_KEY = b"thisis16byteskey"  # 16 bytes
aesgcm = AESGCM(AES_KEY)
client = socket.socket()
client.connect(('localhost', 8888))

client.send(b"johndoe")
ack = client.recv(1024)
client.send(b"john@123")

if client.recv(1024) != b"AUTH_OK":
    print("Login failed.")
    client.close()
    exit()

print("Logged in. Ready to send file.")

filename = "sample.txt"
with open(filename, 'rb') as f:
    file_data = f.read()

nonce = os.urandom(12)
encrypted_data = aesgcm.encrypt(nonce, file_data, None)

# Send nonce
client.send(nonce)

# Send metadata as JSON
metadata = {
    "filename": filename,
    "file_size": len(encrypted_data)
}
metadata_json = json.dumps(metadata).encode()
client.send(f"{len(metadata_json):04}".encode())  # Send metadata length first (4-byte)
client.send(metadata_json)

# Send encrypted file
client.sendall(encrypted_data)

print(" Server:", client.recv(1024).decode())
client.close()
