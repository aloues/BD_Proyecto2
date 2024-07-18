from collections import Counter

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from rtree import index
from tqdm import tqdm


class ImageProcessor:
  def __init__(self, dataset_path, image_size):
    self.dataset_path = dataset_path
    self.sift = cv2.SIFT_create()
    self.image_size = image_size

  def get_image_path(self, filename):
    return f'{self.dataset_path}/images/{filename}'

  def get_image_id(self, filename):
    return int(filename.split('.')[0])

  def get_filename_from_image_id(self, image_id):
    return f'{image_id}.jpg'

  def load_and_process(self, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, self.image_size)
    return image

  def extract_sift_features(self, image, transform=None):
    keypoints, descriptors = self.sift.detectAndCompute(image, None)
    
    if transform and descriptors is not None:
      descriptors = transform(descriptors)
  
    return keypoints, descriptors
  
  def get_image_descriptors(self, filename, transform=None):
    image_path = self.get_image_path(filename)

    try:
      image = self.load_and_process(image_path)
    except Exception:
      return None

    keypoints, descriptors = self.extract_sift_features(image, transform)

    return descriptors

class KnnRTree:
  def __init__(self, index_path, dataset_path, image_size, dataset_size=-1, dimension=128):
    self.df = pd.read_csv(f'{dataset_path}/images.csv')
    self.dataset_size = (len(self.df) if dataset_size == -1 or dataset_size > len(self.df) else dataset_size)

    self.image_processor = ImageProcessor(dataset_path, image_size)
    
    p = index.Property()
    p.dimension = dimension
    self.idx = index.Index(index_path, properties=p)

  def index_descriptors(self, image_id, descriptors):
    if descriptors is None:
      return

    for descriptor in descriptors:
      self.idx.insert(image_id, descriptor)

  def build(self, transform=None):
    count = 0

    for i, row in tqdm(self.df.iterrows(), total=self.dataset_size, desc='KnnRTree (build): '):
      if self.dataset_size > 0 and count >= self.dataset_size:
        break

      filename = row['filename']
      
      descriptors = self.image_processor.get_image_descriptors(filename, transform)
      image_id = self.image_processor.get_image_id(filename)
      self.index_descriptors(image_id, descriptors)
      
      count += 1

  def load_image_to_ax(self, image_path, ax, title=None):
    image = cv2.imread(image_path)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    if title:
      ax.set_title(title)

  def visualize_query(self, image_path, top_k_images):
    fig, axs = plt.subplots(1, len(top_k_images) + 1, figsize=(20, 10))

    self.load_image_to_ax(image_path, axs[0], 'Query Image')

    for i, (image_id, score) in enumerate(top_k_images):
      filename = self.image_processor.get_filename_from_image_id(image_id)
      image_path = self.image_processor.get_image_path(filename)

      self.load_image_to_ax(image_path, axs[i + 1], title=f'K{i + 1} - Score: {score:.2f}')

    plt.show()

  def knn_search_w_score(self, descriptor, k):
    nearest = list(self.idx.nearest(descriptor, k))
    score_results = []

    # Asignar pesos decrecientes a las posiciones
    for i, image_id in enumerate(nearest):
      weight = 1.0 / (i + 1)
      score_results.append((image_id, weight))

    return score_results

  def query(self, image, transform=None, k=8):
    keypoints, descriptors = self.image_processor.extract_sift_features(image, transform)
    image_scores = Counter()

    for descriptor in tqdm(descriptors, total=len(descriptors), desc='KnnRTree (query): '):
      score_results = self.knn_search_w_score(descriptor, k)

      for image_id, weight in score_results:
        image_scores[image_id] += weight

    top_k_images = image_scores.most_common(k)

    return top_k_images