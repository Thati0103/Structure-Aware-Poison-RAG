"""Store PDF pages and their ColQwen2 embeddings in PostgreSQL."""
import os

import pymupdf
import torch
from psycopg2.extras import execute_values

import database
import retrieve

# Indexing and retrieval use the same encoder and processor.
setup = retrieve.setup
_settings = {}


def configure(settings):
    global _settings
    _settings = dict(settings)
    retrieve.configure(settings)


def embed_page(page):
    model, processor = setup()
    image = page.get_pixmap(
        dpi=_settings.get("dpi", 150), alpha=False
    ).pil_image().convert("RGB")

    inputs = processor.process_images([image]).to(model.device)

    with torch.inference_mode():
        embeddings = model(**inputs)[0]

    return embeddings.float().cpu().tolist()

def insert_pages(curr, document_id, filepath, ispoisoned, is_update):
    with pymupdf.open(filepath) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            embeddings = embed_page(page)
            text = page.get_text().replace("\x00", "")
            if is_update:
                curr.execute(
                    "UPDATE pages SET text = %s, ispoisoned = %s WHERE docid = %s AND page_num = %s returning id",
                    (text, ispoisoned, document_id, page_num),
                )
                result = curr.fetchone()
                if result is None:
                    curr.execute("INSERT INTO pages (docid, page_num, text, ispoisoned) VALUES (%s, %s, %s, %s) RETURNING id",
                                 (document_id, page_num, text, ispoisoned))
                    result = curr.fetchone()
                page_id = result[0]
            else:
                curr.execute(
                    "INSERT INTO pages (docid, page_num, text, ispoisoned) "
                    "VALUES (%s, %s, %s, %s) RETURNING id ",
                    (document_id, page_num, text, ispoisoned),
                )
                page_id = curr.fetchone()[0]
            rows = [(page_id, position, str(vector)) for position, vector in enumerate(embeddings)]
            execute_values(
                curr,
                "INSERT INTO page_embeddings (page_id, vector_position, embedding) VALUES %s on conflict (page_id, vector_position) do update set embedding = excluded.embedding",
                rows,
                template="(%s, %s, %s::vector)",
                page_size=500,
            )
            if is_update:
                curr.execute(
                    "DELETE FROM page_embeddings WHERE page_id = %s AND vector_position >= %s",
                    (page_id, len(embeddings)),
                )

def insert_document(filepath, filename, ispoisoned):
    status = False
    document_id = None
    is_update = False
    # connect to the postgresql database
    conn = database.create_connection()
    try:
        with conn.cursor() as curr:
            # first check if document already exists in the database
            curr.execute("SELECT * FROM documents WHERE name = %s and ispoisoned = %s and filepath = %s", (filename, ispoisoned, filepath))
            result = curr.fetchone()
            if result is None:
                # insert document metadata into documents table
                curr.execute(
                    "INSERT INTO documents (version, name, ispoisoned, filepath) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (1, filename, ispoisoned, filepath),
                )
                document_id = curr.fetchone()[0]
            
            else:
                # update the version of the document if it already exists
                document_id = result[0]
                curr.execute(
                    "UPDATE documents SET version = version + 1 where id = %s", (document_id, )
                )
                is_update = True

            insert_pages(curr, document_id, filepath, ispoisoned, is_update)
            with pymupdf.open(filepath) as pdf:
                page_count = len(pdf)
            curr.execute("DELETE FROM page_embeddings WHERE page_id IN (SELECT id FROM pages WHERE docid = %s AND page_num > %s)",
                         (document_id, page_count))
            curr.execute("DELETE FROM pages WHERE docid = %s AND page_num > %s", (document_id, page_count))
            conn.commit()
            status = True
    except Exception as e:
        print(f"Error inserting data for {os.path.basename(filepath)}: {e}")
        conn.rollback()
    finally:
        conn.close()

    return status, document_id if status else None

