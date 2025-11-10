import hashlib

message = "Hello Security Students".encode()
hash_digest = hashlib.sha256(message).hexdigest()

print("SHA256 Hash:", hash_digest)
