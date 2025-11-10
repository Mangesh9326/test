from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

with open("receiver_private_key.pem", "rb") as f:
    receiver_private_key = serialization.load_pem_private_key(f.read(),password=None)

with open("sender_public_key.pem", "rb") as f:
    sender_public_key = serialization.load_pem_public_key(f.read())

with open("encrypted_message.bin", "rb") as f:
    encrypted_message = f.read()

with open("signature.bin", "rb") as f:
    signature = f.read()
    decrypted_message = receiver_private_key.decrypt(encrypted_message, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(), label=None)
)

    print(" Decrypted Message:", decrypted_message.decode())

    try:
        sender_public_key.verify(
        signature,
        decrypted_message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
        )
        print(" Signature Verified. Message is authentic.")
    except Exception as e:
        print(" Signature verification failed:", e)

