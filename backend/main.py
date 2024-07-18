import base64
import json
import time

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bsbi import BSBI
from highD import HighD
from knnRTree import KnnRTree
from knnSequential import KnnSequential
from spotify import get_token, get_track_info, simplify_track_info


class SongsSearchQuery(BaseModel):
    query: str
    k: int = 10
    language: str = "spanish"
    use_postgres: bool = False

class ImagesSearchQuery(BaseModel):
    query: str
    k: int = 8
    n: int = 32000
    model: str

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
data_path = './Data/spotify-dataset/spotify_songs.csv'

index_dir = 'bsbi_index_es'
language = 'spanish'
lang_code = 'es'

bsbi_es = BSBI(initial_block_size, block_size, data_path, language, lang_code, index_dir)

index_dir = 'bsbi_index_en'
language = 'english'
lang_code = 'en'

bsbi_en = BSBI(initial_block_size, block_size, data_path, language, lang_code, index_dir)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return (result, end - start)
    return wrapper

@app.post("/songs/search")
def search(query: SongsSearchQuery):
  execution_time_ms = 0

  if query.use_postgres:
    if query.language == "spanish":  
      pass
    else:
      pass
  else:
    if query.language == "spanish":  
      (results, execution_time_ms) = measure_time(bsbi_es.retrieval)(query.query, query.k)
    else:
      (results, execution_time_ms) = measure_time(bsbi_en.retrieval)(query.query, query.k)

  songs = []

  for track_id, score in results:
    track_info = get_track_info(token, track_id)
    simplified_info = simplify_track_info(track_info)
    simplified_info['id'] = track_id
    simplified_info['score'] = score
    songs.append(simplified_info)
  
  return {"songs": songs, "executionTime": execution_time_ms}

images_dataset_path = './Data/fashion-dataset'
image_size = (64, 64)
n = [1000, 2000, 4000, 8000, 16000, 32000, 44000]
n_components = [8, 16, 32]

models = {
  "knnSequential": {},
  "knnRTree": {},
  "knnDHigh": {8: {}, 16: {}, 32: {}}
}

for size in n:
  models["knnSequential"][size] = KnnSequential(images_dataset_path, image_size, dataset_size=size)

  rTree_index_path = f'./rtree_index/rtree_index_{size}'
  models["knnRTree"][size] = KnnRTree(rTree_index_path, images_dataset_path, image_size, dataset_size=size)

  for n_comp in n_components:
    highD_index_path = f'./highD_index/highD_index_{n_comp}_{size}'
    pca_path = f'./highD_index/pca_{n_comp}_{size}.pkl'
    models["knnDHigh"][n_comp][size] = HighD(highD_index_path, images_dataset_path, image_size, pca_path, n_comp, dataset_size=size)

@app.post("/images/search")
def search(query: ImagesSearchQuery):
  def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
   image = cv2.resize(image, image_size)
   return image
  
  execution_time_ms = 0
  image = readb64(query.query)

  if query.model == "knnSequential":
    (results, execution_time_ms) = measure_time(models["knnSequential"][query.n].query)(image, k=query.k)
  elif query.model == "knnRTree":
    (results, execution_time_ms) = measure_time(models["knnRTree"][query.n].query)(image, k=query.k)
  elif query.model == "highD32":
    (results, execution_time_ms) = measure_time(models["knnDHigh"][32][query.n].query)(image, k=query.k)
  elif query.model == "highD8":
    (results, execution_time_ms) = measure_time(models["knnDHigh"][8][query.n].query)(image, k=query.k)
  else:
    (results, execution_time_ms) = measure_time(models["knnDHigh"][16][query.n].query)(image, k=query.k)


  # Read image metadata
  # Extract cod, name, price and image url
  # Return a list of dictionaries with the image metadata
  images_results = []

  for i, (image_id, score) in enumerate(results):
    try: 
      with open(f'{images_dataset_path}/styles/{image_id}.json', 'r') as file:
        image_style = json.load(file)
        data = image_style['data']
        images_results.append({
          "id": data['id'],
          "name": data['productDisplayName'],
          "price": data['price'],
          "url": data['styleImages']['search']['resolutions']['125X161Xmini'],
          "score": score,
          "variantName": data['variantName'],
          "brandName": data['brandName'],
        })

      file.close()
    except:
      images_results.append({
        "id": image_id,
        "name": f'Image {image_id}',
        "price": 0,
        "url": "https://www.eclosio.ong/wp-content/uploads/2018/08/default.png",
        "score": score,
        "variantName": "Unknown",
        "brandName": "Unknown",
      })

  return {"images": images_results, "executionTime": execution_time_ms}
