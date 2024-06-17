import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

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
initial_block_size = 100 # Number of documents for each initial local index
block_size = 1000  # Number of documents for each block
data_path = './Data/spotify_songs.csv'

index_dir = 'bsbi_index_es'
language = 'spanish'
lang_code = 'es'

bsbi_es = BSBI(initial_block_size, block_size, data_path, language, lang_code, index_dir)

index_dir = 'bsbi_index_en'
language = 'english'
lang_code = 'en'

bsbi_en = BSBI(initial_block_size, block_size, data_path, language, lang_code, index_dir)

@app.post("/search")
def search(query: SearchQuery):
  execution_time_ms = 0

  if query.use_postgres:
    if query.language == "spanish":  
      pass
    else:
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