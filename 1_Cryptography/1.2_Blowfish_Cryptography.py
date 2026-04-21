from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import os

def blowfish_encrypt(key, text):
    iv = os.urandom(8)
    cipher = Cipher(algorithms.Blowfish(key), modes.CBC(iv)).encryptor()

    pad = padding.PKCS7(64).padder()
    data = pad.update(text.encode()) + pad.finalize()

    return iv + cipher.update(data) + cipher.finalize()

def blowfish_decrypt(key, data):
    iv, ct = data[:8], data[8:]
    cipher = Cipher(algorithms.Blowfish(key), modes.CBC(iv)).decryptor()

    unpad = padding.PKCS7(64).unpadder()
    pt = cipher.update(ct) + cipher.finalize()

    return (unpad.update(pt) + unpad.finalize()).decode()

# usage
key = os.urandom(16)
msg = "Hello Blowfish!"

enc = blowfish_encrypt(key, msg)
dec = blowfish_decrypt(key, enc)

print("Message:", msg)
print("Encrypted Blowfish:", enc)
print("Decrypted Blowfish:", dec)
