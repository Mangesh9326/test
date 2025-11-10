import hmac, hashlib

key = b"super_secret_key"
msg = b"Authenticate this message"

hmac_digest = hmac.new(key, msg, hashlib.sha256).hexdigest()
print("HMAC:", hmac_digest)
