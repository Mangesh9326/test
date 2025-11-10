import socket
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from user_auth import authenticate_user, init_db

AES_KEY = b"thisis16byteskey"  # 16 bytes
aesgcm = AESGCM(AES_KEY)

def recv_exact(sock, num_bytes):
    data = b''
    while len(data) < num_bytes:
        part = sock.recv(num_bytes - len(data))
        if not part:
            raise ConnectionError("Connection lost")
        data += part
    return data

init_db()
server = socket.socket()
server.bind(('localhost', 8888))
server.listen(1)
print("File Server listening on port 8888...")
conn, addr = server.accept()
print(f"Connected by {addr}")

username = conn.recv(1024).decode()
conn.send(b"SEND_PASSWORD")
password = conn.recv(1024).decode()

if authenticate_user(username, password):
    conn.send(b"AUTH_OK")
    print("Authenticated:", username)

    # Step 1: Receive nonce (12 bytes)
    nonce = recv_exact(conn, 12)

    # Step 2: Receive metadata length (4 bytes)
    metadata_len = int(recv_exact(conn, 4).decode())

    # Step 3: Receive metadata
    metadata_json = recv_exact(conn, metadata_len).decode()
    metadata = json.loads(metadata_json)
    filename = metadata["filename"]
    file_size = metadata["file_size"]

    # Step 4: Receive encrypted file
    encrypted_data = recv_exact(conn, file_size)

    try:
        decrypted = aesgcm.decrypt(nonce, encrypted_data, None)
        with open("received_" + filename, 'wb') as f:
            f.write(decrypted)
        conn.send(b"File received and decrypted.")
        print(f"File saved: received_{filename}")
    except Exception as e:
        print("Decryption failed:", e)
        conn.send(b"File decryption failed.")
else:
    conn.send(b"AUTH_FAILED")
    conn.close()
