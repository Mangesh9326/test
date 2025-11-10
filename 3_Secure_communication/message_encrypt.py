from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization

with open("receiver_public_key.pem", "rb") as f:
    receiver_public_key = serialization.load_pem_public_key(f.read())

with open("sender_private_key.pem", "rb") as f:
    sender_private_key = serialization.load_pem_private_key(f.read(), password=None)
    message = b"This message is to be sent safely"
    encrypted_message = receiver_public_key.encrypt(message,padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(), label=None)
)

signature = sender_private_key.sign(message,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.MAX_LENGTH),hashes.SHA256()
)

with open("encrypted_message.bin", "wb") as f:
    f.write(encrypted_message)

with open("signature.bin", "wb") as f:
    f.write(signature)

print("Message encrypted and signed.")