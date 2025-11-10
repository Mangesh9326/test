import socket
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization, hashes

with open("client_private.pem", "rb") as f:
    private_key = serialization.load_pem_private_key(f.read(), password=None)

client = socket.socket()
client.connect(('localhost', 7777))

challenge = client.recv(1024)

signature = private_key.sign(
    challenge,
    padding.PKCS1v15(),
    hashes.SHA256()
)

client.send(signature)

response = client.recv(1024).decode()
print(" Server:", response)

client.close()

