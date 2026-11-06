
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_config(key, default=None):
    # First: read nested [database] secrets
    try:
        database_secrets = st.secrets.get("database", {})
        value = database_secrets.get(key.lower())

        if value is not None:
            return value
    except Exception:
        pass

    # Second: read root-level Streamlit secrets
    try:
        value = st.secrets.get(key)
        if value is not None:
            return value
    except Exception:
        pass

    # Third: read local environment variables
    return os.getenv(key, default)



DATABASE_MODE = str(
    get_config("DATABASE_MODE", "sqlite")
).strip().lower()

SQLITE_PATH = BASE_DIR / "database" / "helpdesk.db"

DATABASE_URL = str(
    get_config("DATABASE_URL", "")
).strip()



def get_connection():
    """
    Return a database connection based on DATABASE_MODE.
    Supports local SQLite and cloud PostgreSQL.
    """

    if DATABASE_MODE == "sqlite":
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(SQLITE_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    elif DATABASE_MODE == "postgres":
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is missing from the .env file."
            )

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(
                DATABASE_URL,
                cursor_factory=RealDictCursor,
                connect_timeout=15,
                sslmode="require"
            )
            return conn

        except ImportError as exc:
            raise RuntimeError(
                "psycopg2-binary is not installed."
            ) from exc

    else:
        raise ValueError(
            f"Unsupported DATABASE_MODE: {DATABASE_MODE}"
        )


@contextmanager
def database_connection():
    """
    Manage connection lifecycle and transaction handling.
    """

    conn = get_connection()

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def get_database_type():
    return DATABASE_MODE


def test_connection():
    """
    Test the configured database without modifying data.
    """

    conn = get_connection()

    try:
        if DATABASE_MODE == "sqlite":
            result = conn.execute(
                "SELECT sqlite_version()"
            ).fetchone()
            return {
                "status": "SUCCESS",
                "database": "SQLite",
                "version": result[0]
            }

        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        result = cursor.fetchone()

        return {
            "status": "SUCCESS",
            "database": "PostgreSQL",
            "version": result["version"]
        }

    finally:
        conn.close()


def execute_query(conn, sql, params=()):
    """
    Execute parameterized SQL for SQLite or PostgreSQL.
    Returns a cursor.
    """

    if DATABASE_MODE == "sqlite":
        return conn.execute(sql, params)

    cursor = conn.cursor()
    postgres_sql = sql.replace("?", "%s")
    cursor.execute(postgres_sql, params)
    return cursor


def fetch_one(conn, sql, params=()):
    cursor = execute_query(conn, sql, params)
    return cursor.fetchone()


def fetch_all(conn, sql, params=()):
    cursor = execute_query(conn, sql, params)
    return cursor.fetchall()
