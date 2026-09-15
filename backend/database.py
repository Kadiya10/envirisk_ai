import sqlite3
import os


DATABASE_PATH = "data/envirisk.db"


def get_connection():
    os.makedirs(
        "data",
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()

    connection.close()


def create_user(
    name: str,
    email: str,
    password_hash: str
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO users (
            name,
            email,
            password_hash
        )
        VALUES (?, ?, ?)
        """,
        (
            name,
            email,
            password_hash
        )
    )

    connection.commit()

    user_id = cursor.lastrowid

    connection.close()

    return user_id


def get_user_by_email(
    email: str
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            name,
            email,
            password_hash,
            created_at
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    user = cursor.fetchone()

    connection.close()

    return user