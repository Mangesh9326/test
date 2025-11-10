import sqlite3

def init_db():
    conn = sqlite3.connect('cloud_users.db')
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        role TEXT NOT NULL
    )
    """)

    c.execute('INSERT OR IGNORE INTO users VALUES (?, ?)', ("admin", "admin"))
    c.execute('INSERT OR IGNORE INTO users VALUES (?, ?)', ("bob", "viewer"))
    
    conn.commit()
    conn.close()
    
if __name__ == "__main__":
    init_db()
    print(" DB Initialized")