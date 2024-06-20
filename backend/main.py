import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import psycopg2
import csv

from bsbi import BSBI
from spotify import get_token, get_track_info, simplify_track_info

class SearchQuery(BaseModel):
    query: str
    k: int = 10
    language: str = "spanish"
    use_postgres: bool = False

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

token = get_token()

# Initialize BSBI index
initial_block_size = 100  # Number of documents for each initial local index
block_size = 1000  # Number of documents for each block
data_path = './Data/spotify_songs.csv'

index_dir_es = 'bsbi_index_es'
index_dir_en = 'bsbi_index_en'
bsbi_es = BSBI(initial_block_size, block_size, data_path, 'spanish', 'es', index_dir_es)
bsbi_en = BSBI(initial_block_size, block_size, data_path, 'english', 'en', index_dir_en)

# Función de conexión a PostgreSQL
def connect_to_postgres():
    try:
        conn = psycopg2.connect(
            dbname="spotify_db",
            user="postgres",
            password="password",
            host="localhost"
        )
        print("Conexión exitosa a PostgreSQL")
        return conn
    except Exception as e:
        print(f"Error al conectar a PostgreSQL: {e}")
        return None

# conn = connect_to_postgres()

# Crear tabla y poblarla
def create_and_populate_table(conn):
    conn = connect_to_postgres()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            track_id TEXT PRIMARY KEY,
            track_name TEXT,
            track_artist TEXT,
            lyrics TEXT,
            track_popularity INTEGER,
            track_album_id TEXT,
            track_album_name TEXT,
            track_album_release_date DATE,
            playlist_name TEXT,
            playlist_id TEXT,
            playlist_genre TEXT,
            playlist_subgenre TEXT,
            danceability FLOAT,
            energy FLOAT,
            key INTEGER,
            loudness FLOAT,
            mode INTEGER,
            speechiness FLOAT,
            acousticness FLOAT,
            instrumentalness FLOAT,
            liveness FLOAT,
            valence FLOAT,
            tempo FLOAT,
            duration_ms INTEGER,
            language TEXT
        );
        """)
        
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Omitir la cabecera
            for row in reader:
                cursor.execute("""
                INSERT INTO songs (track_id, track_name, track_artist, lyrics, track_popularity, track_album_id, track_album_name, 
                track_album_release_date, playlist_name, playlist_id, playlist_genre, playlist_subgenre, danceability, energy, key, loudness, 
                mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, language) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, row)
        
        conn.commit()
        cursor.close()
        conn.close()

# Crear índices en PostgreSQL
def create_indexes(conn):
    if conn:
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_name ON songs (track_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_track_artist ON songs (track_artist);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lyrics ON songs USING GIN (lyrics gin_trgm_ops);")
        conn.commit()
        cursor.close()
        conn.close()

# create_and_populate_table(conn)
# create_indexes(conn)

# Función para realizar una consulta en PostgreSQL
def query_postgres(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return results

@app.post("/search")
def search(query: SearchQuery):
    execution_time_ms = 0
    results = []

    if query.use_postgres:
        # if conn:
        #     sql_query = "SELECT * FROM songs WHERE track_name ILIKE %s"
        #     start_time = time.time()
        #     results = query_postgres(conn, (f"%{query.query}%",))
        #     execution_time_ms = (time.time() - start_time) * 1000
        #     conn.close()
        pass
    else:
        if query.language == "spanish":
            start_time = time.time()
            results = bsbi_es.retrieval(query.query, query.k)
            execution_time_ms = (time.time() - start_time) * 1000
        else:
            start_time = time.time()
            results = bsbi_en.retrieval(query.query, query.k)
            execution_time_ms = (time.time() - start_time) * 1000

    songs = []

    for track_id, score in results:
        track_info = get_track_info(token, track_id)
        simplified_info = simplify_track_info(track_info)
        simplified_info['id'] = track_id
        simplified_info['score'] = score
        songs.append(simplified_info)

    return {"songs": songs, "executionTime": execution_time_ms}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
