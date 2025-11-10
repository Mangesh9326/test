import sqlite3
import hashlib
import secrets

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users ( username TEXT PRIMARY KEY, salt TEXT NOT NULL, password_hash TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    if not salt:
        salt = secrets.token_hex(16)
    hash_val = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return salt, hash_val.hex()


def register_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    if cursor.fetchone():
        conn.close()
        return "Username already exists."
    salt, password_hash = hash_password(password)
    cursor.execute('INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)',
    (username, salt, password_hash))
    conn.commit()
    conn.close()
    return "User registered successfully."

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT salt, password_hash FROM users WHERE username = ?',
    (username,))
    result = cursor.fetchone()
    conn.close()
    if not result:
        return False
    salt, stored_hash = result
    _, input_hash = hash_password(password, salt)
    return input_hash == stored_hash

if __name__ == "__main__":
    init_db()
    print(register_user("johndoe", "john@123"))
    print(register_user("johndoe", "anotherpass"))
    print("Login success?" , authenticate_user("johndoe", "john@123"))
    print("Login success?" , authenticate_user("alice", "wrongpass"))