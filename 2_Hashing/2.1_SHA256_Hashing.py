import hashlib

def generate_hash(data):
    hash_object = hashlib.sha256()
    hash_object.update(data.encode())
    return hash_object.hexdigest()

def verify_hash(data, original_hash):
    new_hash = generate_hash(data)
    return new_hash == original_hash

data = "This is a secret message."
hash_value = generate_hash(data)

print(f"Hash: {hash_value}")

print("Verification (correct data):", verify_hash(data, hash_value))

print(
    "Verification (tampered data):",
    verify_hash("This is a tampered message.", hash_value)
)
