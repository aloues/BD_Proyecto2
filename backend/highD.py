import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from knnRTree import KnnRTree


class PCA:
    def __init__(self, dataset_path, dataset_size, batch_size, pca_path, n_components):
      self.df = pd.read_csv(f'{dataset_path}/images.csv')
      self.dataset_size = dataset_size
      self.batch_size = batch_size
      self.pca_path = pca_path

      if os.path.exists(self.pca_path):
        self.ipca = joblib.load(self.pca_path)
        print("Loaded PCA model from disk.")
      else:
        self.ipca = IncrementalPCA(n_components=n_components)


    def build_pca(self, image_processor):
      if os.path.exists(self.pca_path):
        print("PCA model already exists.")
        return

      count = 0
      batch_descriptors = []

      for i, row in tqdm(self.df.iterrows(), total=self.dataset_size, desc='PCA (build): '):
        if self.dataset_size > 0 and count >= self.dataset_size:
          break

        filename = row['filename']
        descriptors = image_processor.get_image_descriptors(filename)

        if descriptors is not None:
          batch_descriptors.extend(descriptors)
        
        if len(batch_descriptors) >= self.batch_size:
          self.ipca.partial_fit(batch_descriptors)
          batch_descriptors = []

        count += 1
      
      # Final partial_fit for remaining descriptors
      if len(batch_descriptors) > 0:
        self.ipca.partial_fit(batch_descriptors)

      # Guardar el modelo PCA en disco
      joblib.dump(self.ipca, self.pca_path)
      print("Saved PCA model to disk.")

class HighD:
  def __init__(self, index_path, dataset_path, image_size, pca_path, n_components, dataset_size=-1):
    self.pca = PCA(dataset_path, dataset_size, 1000, pca_path, n_components)
    self.knnRTree = KnnRTree(index_path, dataset_path, image_size, dataset_size=dataset_size, dimension=n_components)

  def transform(self, descriptors):
    return self.pca.ipca.transform(descriptors)

  def build(self):
    self.pca.build_pca(self.knnRTree.image_processor)
    self.knnRTree.build(transform=self.pca.ipca.transform)

  def query(self, image, k=8):
    return self.knnRTree.query(image, transform=self.transform, k=k)
  
  def visualize_query(self, image_path, top_k_images):
    self.knnRTree.visualize_query(image_path, top_k_images)

