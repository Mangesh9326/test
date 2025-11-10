# Securing Database with Input Validation Code:  ' OR '1'='1' --
import sqlite3  
  
conn = sqlite3.connect('students.db')  
cursor = conn.cursor()  
  
cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")  
cursor.execute("INSERT INTO users VALUES ('admin', 'admin123')")  
conn.commit()   
username = input("Enter username: ")  
password = input("Enter password: ")  
  
query = f"SELECT * FROM users WHERE username = '{username}' AND password ='{password}'"  
cursor.execute(query)   
# cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?",(username, password))   
result = cursor.fetchone()  
print(result)  
if result:  
    print("Login successful!")  
else:  
    print("Invalid credentials.")  





# Encrypting Sensitive Data in Database
import sqlite3  
import bcrypt  
  
conn = sqlite3.connect('secure_users.db')  
cursor = conn.cursor()  
cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")  
conn.commit()  
  
username = input("Enter username: ")  
raw_password = input("Enter password: ")  
  
hashed = bcrypt.hashpw(raw_password.encode(), bcrypt.gensalt())  
cursor.execute("INSERT INTO users VALUES (?, ?)", (username, hashed))  
conn.commit()  
  
login_user = input("Username to login: ")  
login_pass = input("Password to login: ")  
cursor.execute("SELECT password FROM users WHERE username = ?", (login_user,))  
  
data = cursor.fetchone()  
if data and bcrypt.checkpw(login_pass.encode(), data[0]):  
    print("Login successful.")  
else:  
    print("Login failed.")  




# Role-Based Access Control (RBAC) in Database Code:
import sqlite3  
  
conn = sqlite3.connect('rbac.db')  
cursor = conn.cursor()  
  
cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT,role TEXT)")    
cursor.execute("INSERT INTO users VALUES ('admin', 'admin123', 'admin')")  
cursor.execute("INSERT INTO users VALUES ('john', 'john123', 'student')")  
  
conn.commit()  
username = input("Enter username: ")  
password = input("Enter password: ")  
  
cursor.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))   
user = cursor.fetchone()  
if user:  
    role = user[0]  
    print(f"Login successful. Your role is: {role}")  
  
    if role == 'admin':  
        print("Access granted to view, add, delete, modify records.")     
    elif role == 'student':  
        print("Access granted to view records only.")
    else:  
        print("Limited access.")  
else:  
    print("Invalid credentials.") 




# Encrypting Entire DB Fields Using Fernet Code:
from cryptography.fernet import Fernet  
  
key = Fernet.generate_key()  
cipher = Fernet(key)  
  
data = "Classified military documents"  
enc_data = cipher.encrypt(data.encode())  
  
dec_data = cipher.decrypt(enc_data).decode()
print(f"\nDecrypted:", dec_data, '\n')
