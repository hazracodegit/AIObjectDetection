import os
import sqlite3


database_path = os.path.abspath(os.path.join("instance", "ai_detect.sqlite"))
print("DATABASE DB PATH =", database_path)

with sqlite3.connect(database_path) as conn:
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            objects TEXT NOT NULL,
            object_count INTEGER NOT NULL DEFAULT 0,
            confidence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

print("Database tables are ready.")

