import hmac, hashlib

secret_key = b"shared_secret"
message = b"Important Transaction: $500"

mac = hmac.new(secret_key, message, hashlib.sha256).hexdigest()

print("Message:", message.decode())
print("MAC:", mac)
