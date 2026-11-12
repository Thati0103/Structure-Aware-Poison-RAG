# Take documents as inputs

from functools import lru_cache
import os
import psycopg2
import pymupdf
import database
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from psycopg2.extras import execute_values

@lru_cache(maxsize=1)
    
def setup():
    model_name = "vidore/colqwen2-v1.0"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)
    return model, processor

docs_filepath = "data/docs"

def embed_page(page):
    model, processor = setup()
    image = page.get_pixmap(
        dpi=150, alpha=False
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
                    raise ValueError(f"No page found for document_id {document_id} and page_num {page_num}")
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
            conn.commit()
            status = True
    except psycopg2.DatabaseError as error:
        print(f"Database error for {os.path.basename(filepath)}: {error}")
        conn.rollback()
    except Exception as e:
        print(f"Error inserting data for {os.path.basename(filepath)}: {e}")
        conn.rollback()
    finally:
        conn.close()

    return status, document_id if status else None

def prepare_data(ispoisoned):
    folder = os.path.join(
        docs_filepath,
        "pois" if ispoisoned else "non_pois",
    )

    # read all files in directory for each file 
    # insert file metadata and pages
    for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(".pdf"):
                insert_document(filepath, filename, ispoisoned)
                
def delete_document(filepath, filename):
    # connect to the postgresql database
    conn = database.create_connection()
    try:
        with conn.cursor() as curr:
            # first check if document already exists in the database
            curr.execute("SELECT id FROM documents WHERE name = %s and filepath = %s", (filename, filepath))
            result = curr.fetchone()
            if result is not None:
                document_id = result[0]
                # delete existing pages and embeddings for the document
                curr.execute("DELETE FROM page_embeddings WHERE page_id IN (SELECT id FROM pages WHERE docid = %s)", (document_id,))
                curr.execute("DELETE FROM pages WHERE docid = %s", (document_id,))
                curr.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                conn.commit()
    except psycopg2.DatabaseError as error:
        print(f"Database error for {os.path.basename(filepath)}: {error}")
        conn.rollback()
    except Exception as e:
        print(f"Error deleting data for {os.path.basename(filepath)}: {e}")
        conn.rollback()
    finally:
        conn.close()
        
def delete_data(ispoisoned):
    folder = os.path.join(
        docs_filepath,
        "pois" if ispoisoned else "non_pois",
    )

    # read all files in directory for each file 
    # delete file metadata and pages
    for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(".pdf"):
                delete_document(filepath, filename)