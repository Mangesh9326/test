from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

# generate keys
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

# common padding
oaep = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None
)

# message
message = "Hello RSA!"

# encrypt & decrypt
encrypted = public_key.encrypt(message.encode(), oaep)
decrypted = private_key.decrypt(encrypted, oaep).decode()

# same output format
print("Message:", message)
print("Encrypted RSA:", encrypted)
print("Decrypted RSA:", decrypted)
