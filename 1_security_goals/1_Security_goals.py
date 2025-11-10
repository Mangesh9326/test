# Confidentiality
from cryptography.fernet import Fernet   
key = Fernet.generate_key()  
cipher = Fernet(key)  
message = "This message is to be encoded and decoded using AES encryption."  
encrypted = cipher.encrypt(message.encode())  
decrypted = cipher.decrypt(encrypted).decode() 
print(f"\nKey:", key.decode())  
print(f"Encrypted:", encrypted)  
print(f"Decrypted:", decrypted, "\n") 


# Integrity
import hashlib  
data = "The melancholy of Haruhi Suzumiya"  
hash1 = hashlib.sha256(data.encode()).hexdigest()  
tampered = "The MELANCHOLY of Haruhi Suzumiya"  
hash2 = hashlib.sha256(tampered.encode()).hexdigest() 
print("\nHash1:", hash1)  
print("Hash2:", hash2)  
print(f"Integrity Verified?", hash1 == hash2, "\n") 


# Availability
import shutil  
original_file = 'sample.txt'  
backup_file = 'backup_data.txt'  
shutil.copyfile(original_file, backup_file)  
print("Backup completed!") 


# Authentication
users = {  
'admin': 'admin123',  
'user123': 'password123'  
}  
username = input("\nEnter username: ")  
password = input("Enter password: ")  
if users.get(username) == password:  
    print("Access granted\n")  
else:  
    print("Access denied\n") 


# Non-Repudiation
import hashlib  
message = "The sent message to be signed"  
signature = hashlib.sha256(message.encode()).hexdigest()  
received_message = "The received message to be signed" 
received_signature =  hashlib.sha256(received_message.encode()).hexdigest()  
print(f"\nSignature Verified:", signature == received_signature,'\n') 

# Accountability
import datetime  
  
def log_action(user, action):  
   with open("audit_log.txt", "a") as log:  
       timestamp = datetime.datetime.now()  
       log.write(f"{timestamp} - {user}: {action}\n") 
log_action("admin", "Logged In")  
log_action("user1", "Viewed Report") 
