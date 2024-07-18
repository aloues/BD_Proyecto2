import heapq
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from knnRTree import ImageProcessor


class KnnSequential:
  def __init__(self, dataset_path, image_size, dataset_size=-1):
    self.df = pd.read_csv(f'{dataset_path}/images.csv')
    self.dataset_path = dataset_path
    self.dataset_size = (len(self.df) if dataset_size == -1 or dataset_size > len(self.df) else dataset_size)

    self.image_processor = ImageProcessor(dataset_path, image_size)

  def distance(self, point, query):
    return np.linalg.norm(point[1] - query)
  
  def push_point(self, heap, point, query, k):
    distance = self.distance(point, query)
    if len(heap) < k:
      heapq.heappush(heap, (-distance, point))
    else:
      heapq.heappushpop(heap, (-distance, point))
    
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
  
  def knn_search_w_score(self, query_descriptors, k):
    heaps = []

    for _ in range(len(query_descriptors)):
      heaps.append([])

    count = 0

    for i, row in tqdm(self.df.iterrows(), total=self.dataset_size, desc='KnnSequential (build): '):
      if self.dataset_size > 0 and count >= self.dataset_size:
        break

      filename = row['filename']

      descriptors = self.image_processor.get_image_descriptors(filename)
      image_id = self.image_processor.get_image_id(filename)

      if descriptors is None:
        continue
      
      for descriptor in descriptors:
        for i, query_descriptor in enumerate(query_descriptors):
          self.push_point(heaps[i], (image_id, descriptor), query_descriptor, k)
      
      count += 1

    score_results = Counter()

    for heap in heaps:
      heap = sorted(heap, reverse=True)

      for i, (_, (filename, _)) in enumerate(heap):
        weight = 1.0 / (i + 1)
        score_results[filename] += weight

    return score_results.most_common(k)

  def query(self, image, k=8):
    keypoints, descriptors = self.image_processor.extract_sift_features(image)
    top_k_images = self.knn_search_w_score(descriptors, k)

    return top_k_images
  
