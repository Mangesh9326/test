#User Authentication
import hashlib

users = {}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register(username, password):
    users[username] = hash_password(password)
    print("Registered Successfully!")

def login(username, password):
    if users.get(username) == hash_password(password):
        print("Access Granted!")
    else:
        print("Access Denied!")

register("admin", "securepass")
login("admin", "securepass")
login("admin", "wrongpass")


# Role-Based file Access
permissions = {
    "admin": ["read", "write"],
    "user1": ["read"]
}

def access_file(user_role, action):
    if action in permissions.get(user_role, []):
        print(f"{user_role} can {action} the file")
    else:
        print(f"{user_role} cannot {action} the file")

access_file("admin", "write")
access_file("user1", "write")


# File Encryption with AES
from cryptography.fernet import Fernet

key = Fernet.generate_key()
with open("encrypted.key", "wb") as filekey:
    filekey.write(key)

def encrypt_file(filename):
    with open("encrypted.key", "rb") as filekey:
        key = filekey.read()
        fernet = Fernet(key)
    with open(filename, "rb") as file:
        data = file.read()
        encrypted = fernet.encrypt(data)
    with open(filename, "wb") as file:
        file.write(encrypted)
    print("File encrypted")


# File Integrity Checker
import hashlib   
def get_file_hash(filename):  
   with open(filename, 'rb') as f:  
       return hashlib.sha256(f.read()).hexdigest()  
    
original_hash = get_file_hash("sample.txt")  
current_hash = get_file_hash("sample.txt")   
if original_hash == current_hash:  
   print(" File is intact")  
else:  
   print(" File has been changed!")


# Basic IP Firewall Simulation
blocked_ips = ["192.168.1.100", "10.0.0.5"]

def check_access(ip):
    if ip in blocked_ips:
        print(f"Access denied for {ip}")
    else:
        print(f"Access granted for {ip}")

check_access("192.168.1.100")
check_access("8.8.8.8")


# Activity Logging
import datetime

def log_activity(user, action):
    with open("access.log", "a") as logfile:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logfile.write(f"[{time}] {user} -> {action}\n")

log_activity("admin", "Accessed Settings")
log_activity("guest", "Tried to Delete File")

print("File Created")
