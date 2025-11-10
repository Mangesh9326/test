import hmac
import hashlib

secret_key = b"iot_secret_key"

message = b"Sigma: 100%"

hmac_digest = hmac.new(secret_key, message, hashlib.sha256).hexdigest()
print(f"\nMessage: {message.decode()}")
print(f"Generated HMAC: {hmac_digest}")

received_hmac = hmac.new(secret_key, message, hashlib.sha256).hexdigest()

if hmac.compare_digest(hmac_digest, received_hmac):
    print(f"Message is authentic\n")
else:
    print(f"Message tampered!\n")