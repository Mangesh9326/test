from Crypto.Cipher import AES

import base64

key = b"ThisIsA16ByteKey"
cipher = AES.new(key, AES.MODE_EAX)

sensor_data = "Air Quality Index: 30"

ciphertext, tag = cipher.encrypt_and_digest(sensor_data.encode())
nonce = cipher.nonce
print(f"\nEncrypted Data: {base64.b64encode(ciphertext).decode()}")

cipher_dec = AES.new(key, AES.MODE_EAX, nonce=nonce)
decrypted_data = cipher_dec.decrypt(ciphertext).decode()
print(f"Decrypted Data: {decrypted_data}\n")
