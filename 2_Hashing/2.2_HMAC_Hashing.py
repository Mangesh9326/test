import hmac
import hashlib

def generate_hash(data):
    hash_object = hashlib.sha256()
    hash_object.update(data.encode())
    return hash_object.hexdigest()

def verify_hash(data, original_hash):
    new_hash = generate_hash(data)
    return hmac.compare_digest(new_hash, original_hash)

def check_data_integrity(data, original_hash):
    if verify_hash(data, original_hash):
        print("Data is intact and verified.")
    else:
        print("Warning: Data has been tampered with!")

original_data = "This is a secret message."
hash_value = generate_hash(original_data)

print(f"Original Hash: {hash_value}\n")

print("Checking original data:")
check_data_integrity(original_data, hash_value)

tampered_data = "This is a hacked message."
print("\nChecking tampered data:")
check_data_integrity(tampered_data, hash_value)

tampered_data_space = "This is a secret message. "
print("\nChecking data with extra space:")
check_data_integrity(tampered_data_space, hash_value)

tampered_data_char = "This is a secrt message."
print("\nChecking data with missing character:")
check_data_integrity(tampered_data_char, hash_value)
