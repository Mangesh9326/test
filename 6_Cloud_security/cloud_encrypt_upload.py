import boto3
from cryptography.fernet import Fernet
import sqlite3
import os

def get_user_role(username):
   conn = sqlite3.connect('cloud_users.db')
   c = conn.cursor()
   c.execute('SELECT role FROM users WHERE username=?', (username,))
   result = c.fetchone()
   conn.close()
   return result[0] if result else None

def encrypt_file(filename, key_file="aes.key"):
   with open(key_file, "rb") as f:
       key = f.read()
   fernet = Fernet(key)
   with open(filename, "rb") as file:
       original = file.read()
   encrypted = fernet.encrypt(original)
   with open("enc_" + filename, "wb") as enc_file:
       enc_file.write(encrypted)
   return "enc_" + filename

def upload_to_s3(file_path, bucket_name="cloud-sec-bucket"):
   try:
       if not os.path.exists(file_path):
           print(f"Error: File {file_path} not found")
           return False
      
       file_size = os.path.getsize(file_path)
       print(f" Simulating upload of {file_path} ({file_size} bytes) to bucket {bucket_name}")
       print(f" Upload successful! (Simulated)")
       print(f" File would be accessible at: s3://{bucket_name}/{file_path}")
       return True
      
   except Exception as e:
       print(f"Upload simulation failed: {e}")
       return False
# --- Main logic ---
if __name__ == "__main__":
   username = input(" Enter your username: ")
   role = get_user_role(username)
   if role != "admin":
       print("Access Denied. Only admin can upload.")
   else:
       file_to_upload = input(" Enter file name to upload: ")
       enc_file = encrypt_file(file_to_upload)
       upload_to_s3(enc_file)

