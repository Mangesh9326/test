#pip install cryptography
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms, modes 
from cryptography.hazmat.primitives import padding 
from cryptography.hazmat.backends import default_backend 
import os 
 
def aes_encrypt(key, plaintext): 
    iv = os.urandom(16)  
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), 
    backend=default_backend()) 
    encryptor = cipher.encryptor() 
    padder = padding.PKCS7(128).padder() 
    padded_data = padder.update(plaintext.encode()) + padder.finalize()  
    ciphertext = encryptor.update(padded_data) + encryptor.finalize() 
    return iv + ciphertext # prepend IV for use in decryption 
 
def aes_decrypt(key, ciphertext): 
    iv = ciphertext[:16] 
    actual_ciphertext = ciphertext[16:] 
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), 
    backend=default_backend()) 
    decryptor = cipher.decryptor() 
    padded_plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize() 
    unpadder = padding.PKCS7(128).unpadder() 
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize() 
    return plaintext.decode() 
 
key = os.urandom(32)  
message = "Hello AES!" 
encrypted = aes_encrypt(key, message) 
print("Message: ",message) 
print("Encrypted AES:", encrypted) 
print("Decrypted AES:", aes_decrypt(key, encrypted)) 
