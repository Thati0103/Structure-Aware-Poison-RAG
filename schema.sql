CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY  PRIMARY KEY,
    version INTEGER NOT NULL check (version > 0),
    NAME VARCHAR(255) NOT NULL,
    ispoisoned BOOLEAN NOT NULL,
    filepath VARCHAR(255) NOT NULL,
    unique(NAME)
);

CREATE TABLE IF NOT EXISTS pages (
    id BIGINT GENERATED ALWAYS AS IDENTITY  PRIMARY KEY,
    docid BIGINT NOT NULL references documents(id),
    page_num INTEGER NOT NULL check (page_num > 0),
    text TEXT NOT NULL,
    ispoisoned BOOLEAN NOT NULL,
    unique(docid, page_num)
);
create table if not exists page_embeddings (
    page_id BIGINT NOT NULL references pages(id),
    embedding VECTOR NOT NULL,
    vector_position Integer NOT NULL check (vector_position >= 0),
    primary key(page_id, vector_position)
);

CREATE TABLE IF NOT EXISTS questions (
    id BIGINT GENERATED ALWAYS AS IDENTITY  PRIMARY KEY,
    question_content TEXT NOT NULL,
    answer TEXT NOT NULL,
    page_ids BIGINT[] NOT NULL
);

Create table if not exists question_embeddings (
    question_id BIGINT NOT NULL references questions(id),
    embedding VECTOR NOT NULL,
    vector_position Integer NOT NULL check (vector_position >= 0),
    primary key(question_id, vector_position)
);
