import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "movie.db")

_MIGRATED = False


def get_db():
    global _MIGRATED
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 幂等建表：云端首次启动时数据库为空，自动创建完整表结构（只执行一次）
    if not _MIGRATED:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                security_answer TEXT NOT NULL,
                login_attempts INTEGER DEFAULT 0,
                locked_until REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                UNIQUE(user_id, movie_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                UNIQUE(user_id, movie_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dislikes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                UNIQUE(user_id, movie_id)
            )
        """)
        conn.commit()

        # 兼容旧数据库：缺少安全字段时自动补齐
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        if "login_attempts" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN login_attempts INTEGER DEFAULT 0")
        if "locked_until" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN locked_until REAL DEFAULT 0")

        # 反馈闭环：用户标记“不感兴趣”的电影
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dislikes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                UNIQUE(user_id, movie_id)
            )
        """)
        conn.commit()
        _MIGRATED = True

    return conn
