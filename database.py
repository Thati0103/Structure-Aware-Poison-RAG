# create schema in database

import psycopg2

def create_connection():
    # connect to the postgresql database
    conn = psycopg2.connect(database="postgres", user="ts", host="localhost")
    return conn


def create_schema():
    # Read the schema file
    filepath = "schema.sql"
    with open (filepath, encoding='utf-8') as file:
        content = file.read()
    
    # connect to the postgresql database
    conn = create_connection()
    with conn.cursor() as curr:
        try:
            curr.execute(content)
        except psycopg2.DatabaseError as error:
            print(error)

    conn.commit()
    conn.close()

def drop_database():
    conn = create_connection()
    with conn.cursor() as curr:
        curr.execute("""
            DROP TABLE IF EXISTS question_embeddings;
            DROP TABLE IF EXISTS page_embeddings;
            DROP TABLE IF EXISTS questions;
            DROP TABLE IF EXISTS pages;
            DROP TABLE IF EXISTS documents;
        """)

    conn.commit()
    conn.close()

create_schema()
