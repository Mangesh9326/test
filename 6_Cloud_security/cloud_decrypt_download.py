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

def download_from_s3(file_name, bucket_name="cloud-sec-bucket"):
   try:
       if os.path.exists(file_name):
           print(f" File {file_name} already exists locally")
           print(f" Simulating download of {file_name} from bucket {bucket_name}")
           print(f" Download successful! (Simulated)")
           return True
       else:
           print(f" Error: File {file_name} not found locally")
           print(f" In a real scenario, this would be downloaded from s3://{bucket_name}/{file_name}")
           return False
          
   except Exception as e:
       print(f"Download simulation failed: {e}")
       return False
  
def decrypt_file(enc_file, key_file="aes.key"):
   with open(key_file, "rb") as f:
       key = f.read()
   fernet = Fernet(key)
   with open(enc_file, "rb") as f:
       encrypted_data = f.read()
   decrypted = fernet.decrypt(encrypted_data)
   out_file = "dec_" + enc_file.replace("enc_", "")
   with open(out_file, "wb") as f:
       f.write(decrypted)
   print(f" Decrypted file saved as: {out_file}")

if __name__ == "__main__":
   username = input(" Enter your username: ")
   role = get_user_role(username)
   if role not in ("admin", "viewer"):
       print(" Unauthorized user.")
   else:
       file_name = input(" Enter file to download (e.g., enc_sample.txt): ")
       if download_from_s3(file_name):
           decrypt_file(file_name)
       else:
           print(" Download failed. Cannot proceed with decryption.")