"""Existing PostgreSQL storage, with an isolated schema for experiments."""
import os
import re
from pathlib import Path

import psycopg2
from psycopg2 import sql

_settings = {}


def configure(settings):
    global _settings
    schema = settings.get("schema", "public")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError("Invalid PostgreSQL schema name")
    _settings = dict(settings)


def create_connection():
    options = {
        "database": _settings.get("database", os.environ.get("PGDATABASE", "postgres")),
        "user": _settings.get("user", os.environ.get("PGUSER", "ts")),
        "host": _settings.get("host", os.environ.get("PGHOST", "localhost")),
        "port": _settings.get("port", os.environ.get("PGPORT", "5432")),
    }
    password = os.environ.get(_settings.get("password_env", "PGPASSWORD"))
    if password:
        options["password"] = password
    schema = _settings.get("schema", "public")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError("Invalid PostgreSQL schema name")
    options["options"] = f'-c search_path={schema},public'
    return psycopg2.connect(**options)


def create_schema():
    conn = create_connection()
    try:
        with conn:
            with conn.cursor() as curr:
                schema = _settings.get("schema", "public")
                curr.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
                curr.execute((Path(__file__).parent / "schema.sql").read_text(encoding="utf-8"))
    finally:
        conn.close()

