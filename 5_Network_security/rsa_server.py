import socket
import os
import secrets
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization, hashes

with open("client_public.pem", "rb") as f:
    client_public_key = serialization.load_pem_public_key(f.read())

server = socket.socket()
server.bind(('localhost', 7777))
server.listen(1)
print(" RSA Server running...")

conn, addr = server.accept()

print(" Connected by", addr)

challenge = secrets.token_bytes(32)
conn.send(challenge)

signature = conn.recv(256)
try:
    client_public_key.verify(
    signature,
    challenge,
    padding.PKCS1v15(),
    hashes.SHA256()
    )
    conn.send(b"AUTH_OK")
    print(" RSA Authentication Successful")
except Exception as e:
    conn.send(b"AUTH_FAILED")
    print(" RSA Authentication Failed:", str(e))

conn.close()