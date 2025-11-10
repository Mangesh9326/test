from cryptography.fernet import Fernet
import os
import shutil

# Create a simulated cloud folder
os.makedirs("cloud_backup", exist_ok=True)

# Step 1: Generate a secret key (AES)
key = Fernet.generate_key()

fernet = Fernet(key)

with open("aes.key", "wb") as kf:
    kf.write(key)

# Step 2: Encrypt the file
def encrypt_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    encrypted_data = fernet.encrypt(data)

    encrypted_file_path = "enc_" + os.path.basename(file_path)
    with open(encrypted_file_path, "wb") as ef:
        ef.write(encrypted_data)

    # Simulate cloud upload (just copying)
    shutil.copy(encrypted_file_path, os.path.join("cloud_backup", encrypted_file_path))
    print(f"Encrypted and uploaded: {encrypted_file_path}")


if __name__ == "__main__":
    input_file = "sample2.txt"

    if not os.path.exists(input_file):
        with open(input_file, "w") as f:
            f.write("Confidential Cloud Backup File")

    encrypt_file(input_file)