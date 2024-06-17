import os
import math
import re
import pickle
import nltk
import csv
import shutil

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class InvertIndex:
  def __init__(self, index_file, lang):
    self.index_file = index_file
    self.index = {}
    self.idf = {}
    self.length = {}
    self.stemmer = SnowballStemmer(lang)
    self.stopwords = stopwords.words(lang)

  def preprocesamiento(self, texto):
      tokens = re.findall(r'\b\w+\b', texto.lower())
      tokens = [word for word in tokens if word not in self.stopwords]
      words = [self.stemmer.stem(word) for word in tokens]
      return words

  def calculate_tf(self, doc):
    tf_dict = {}
    for word in doc:
      tf_dict[word] = tf_dict.get(word, 0) + 1

    for word, count in tf_dict.items():
      tf_dict[word] = count / len(doc)
    return tf_dict
      
  def calculate_idf(self, collection):
    df_dict = {}
    for doc in collection:
      for word in doc:
        df_dict[word] = df_dict.get(word, 0) + 1
    
    for word, count in df_dict.items():
      self.idf[word] = math.log(len(collection) / count)

  def building(self, data):
      # build the inverted index with the collection
      processed_collection = {}
      for doc_id, text in data.items():
        words = self.preprocesamiento(text)
        processed_collection[doc_id] = words
  
      for doc_id, words in processed_collection.items():  
        # compute the tf
        tf = self.calculate_tf(words)

        for word, tf_value in tf.items():
          if word not in self.index:
            self.index[word] = []
          self.index[word].append((doc_id, tf_value))
        
        # compute the length (norm)
        self.length[doc_id] = math.sqrt(sum(tf_value**2 for tf_value in tf.values()))

      # compute the idf
      self.calculate_idf(processed_collection.values())

      # sort the index by term
      self.index = dict(sorted(self.index.items()))

      # store in disk
      with open(self.index_file, 'wb') as f:
        pickle.dump((self.index, self.idf, self.length), f)

class BSBI:
  def __init__(self, initial_block_size, block_size, data_path, lang, lang_abbr, index_dir, rebuild=False):
      self.initial_block_size = initial_block_size
      self.block_size = block_size
      self.index_dir = index_dir
      self.data_path = data_path
      self.lang = lang
      self.lang_abbr = lang_abbr
      self.block_filenames = []
      self.stemmer = SnowballStemmer(lang)
      self.stopwords = stopwords.words(lang)

      if not os.path.exists(self.index_dir):
        os.makedirs(self.index_dir)
        self.build_index()
      else:
        if rebuild:
          self.clean_dir(self.index_dir)
          self.build_index()
        else:
          self.load_index()

  def preprocesamiento(self, texto):
      tokens = re.findall(r'\b\w+\b', texto.lower())
      tokens = [word for word in tokens if word not in self.stopwords]
      words = [self.stemmer.stem(word) for word in tokens]
      return words

  def calculate_tf(self, doc):
    tf_dict = {}
    for word in doc:
      tf_dict[word] = tf_dict.get(word, 0) + 1

    for word, count in tf_dict.items():
      tf_dict[word] = count / len(doc)
    return tf_dict

  def store_index(self, index_file, index, idf, length):
    with open(index_file, 'wb') as f:
      pickle.dump((index, idf, length), f)

  def build_local_indexes(self):
    print("Building local indexes")
    block_num = 0
    docs = {}

    with open(self.data_path, 'r') as f:
      reader_obj = csv.reader(f) 

      for row in reader_obj:
        if(row[-1] != self.lang_abbr):
          continue

        docs[row[0]] = row[3]

        if len(docs) == self.initial_block_size:
          block_index_file = os.path.join(self.index_dir, f'block_{block_num}.pkl')
          invert_index = InvertIndex(block_index_file, self.lang)
          invert_index.building(docs)
          self.block_filenames.append(block_index_file)
          docs = {}
          block_num += 1

      # Process the remaining documents in the last block
      if docs:
        block_index_file = os.path.join(self.index_dir, f'block_{block_num}.pkl')
        invert_index = InvertIndex(block_index_file, self.lang)
        invert_index.building(docs)
        self.block_filenames.append(block_index_file)

      print("Local indexes built")

  def merge_two_sorted_arrays(self, arr1, arr2):
    merged_arr = []
    i = j = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i][0] < arr2[j][0]:
            merged_arr.append(arr1[i])
            i += 1
        elif arr1[i][0] > arr2[j][0]:
            merged_arr.append(arr2[j])
            j += 1
        else:
            doc_id = arr1[i][0]
            tf_sum = arr1[i][1] + arr2[j][1]
            merged_arr.append((doc_id, tf_sum))
            i += 1
            j += 1

    while i < len(arr1):
        merged_arr.append(arr1[i])
        i += 1

    while j < len(arr2):
        merged_arr.append(arr2[j])
        j += 1

    return merged_arr

  def load_index_block(self, block_filename):
    with open(block_filename, 'rb') as f:
      index, idf, length = pickle.load(f)
    return index, idf, length

  def merge_two_rows(self, row1, row2, count=0, level=0):
    current_file1_iter = iter(row1)
    current_file2_iter = iter(row2)

    current_file1 = next(current_file1_iter, None)
    current_file2 = next(current_file2_iter, None)

    (index1, idf1, length1) = self.load_index_block(current_file1)
    (index2, idf2, length2) = self.load_index_block(current_file2)

    current_term1_iter = iter(index1.items())
    current_term2_iter = iter(index2.items())

    current_term1 = next(current_term1_iter, None)
    current_term2 = next(current_term2_iter, None)

    merged_files = []
    current_page_index = {}
    current_page_idf = {}
    merged_length = {}
    current_page_size = 0

    for doc_id in set(length1.keys()).union(length2.keys()):
      merged_length[doc_id] = math.sqrt(length1.get(doc_id, 0)**2 + length2.get(doc_id, 0)**2)

    level_dir = os.path.join(self.index_dir, f'level_{level}')
    if not os.path.exists(level_dir):
      os.makedirs(level_dir)

    while current_file1 or current_file2:
      if not current_term1:
        current_file1 = next(current_file1_iter, None)
        if current_file1:
          (index1, idf1, length1) = self.load_index_block(current_file1)
          current_term1_iter = iter(index1.items())
          current_term1 = next(current_term1_iter, None)
      
      if not current_term2:
        current_file2 = next(current_file2_iter, None)
        if current_file2:
          (index2, idf2, length2) = self.load_index_block(current_file2)
          current_term2_iter = iter(index2.items())
          current_term2 = next(current_term2_iter, None)

      if current_term1 and (not current_term2 or current_term1[0] < current_term2[0]):
        term, postings = current_term1
        current_page_index[term] = postings
        current_page_idf[term] = idf1[term]
        current_term1 = next(current_term1_iter, None)
      elif current_term2 and (not current_term1 or current_term2[0] < current_term1[0]):
        term, postings = current_term2
        current_page_index[term] = postings
        current_page_idf[term] = idf2[term]
        current_term2 = next(current_term2_iter, None)
      elif current_term1 and current_term2:
        term, postings1 = current_term1
        _, postings2 = current_term2
        current_page_index[term] = self.merge_two_sorted_arrays(postings1, postings2)
        current_page_idf[term] = idf1[term] + idf2[term]
        current_term1 = next(current_term1_iter, None)
        current_term2 = next(current_term2_iter, None)
      else:
        break

      current_page_size += 1

      if current_page_size >= self.block_size:
        output_file = os.path.join(level_dir, f'block_{count}_{len(merged_files)}.pkl')
        with open(output_file, 'wb') as out:
          pickle.dump((current_page_index, current_page_idf, merged_length), out)
        merged_files.append(output_file)
        current_page_index = {}
        current_page_idf = {}
        current_page_size = 0

    if current_page_index:
      output_file = os.path.join(level_dir, f'block_{count}_{len(merged_files)}.pkl')
      with open(output_file, 'wb') as out:
        pickle.dump((current_page_index, current_page_idf, merged_length), out)
      merged_files.append(output_file)

    return merged_files

  def merge_blocks(self):
    print("Merging blocks")

    filenames = [[filename] for filename in self.block_filenames]
    level = 0

    while len(filenames) > 1:
      print(f"Number of blocks: {len(filenames)}")
      print(filenames)

      new_level = []
      
      for i in range(0, len(filenames), 2):
        if i + 1 < len(filenames):
          merged_files = self.merge_two_rows(filenames[i], filenames[i + 1], i // 2, level)
          new_level.append(merged_files)
        else:
          new_level.append(filenames[i])
      
      filenames = new_level
      level += 1
  
    final_dir = os.path.join(self.index_dir, 'final')
    os.rename(os.path.join(self.index_dir, f'level_{level-1}'), final_dir)
    self.block_filenames = [[f'{final_dir}/{os.path.split(filename)[-1]}' for filename in filenames[0]]]

  def clean_dir(self, dir_path):
    if os.path.exists(dir_path):
      shutil.rmtree(dir_path)
    os.makedirs(dir_path)

  def build_index(self):
    self.build_local_indexes()
    self.merge_blocks()

    # Print the first key of each block
    for block_file in self.block_filenames[0]:
      with open(block_file, 'rb') as f:
        index, idf, length = pickle.load(f)
        keys = list(index.keys())

  def retrieval(self, query, k=10):
    def binary_search_term(blocks, term):
      left, right = 0, len(blocks) - 1

      while left <= right:
        mid = (left + right) // 2
        index_mid, idf_mid, length = self.load_index_block(blocks[mid])
        
        terms_mid = list(index_mid.keys())
        
        if terms_mid[0] <= term <= terms_mid[-1]:
            if term in index_mid:
                tf_values = index_mid[term]
                idf_value = idf_mid.get(term, 0)
                return tf_values, idf_value, length
            else:
                return None
        
        if term < terms_mid[0]:
            right = mid - 1
        else:
            left = mid + 1
      return None
    
    query_words = self.preprocesamiento(query)

    words_tf = {}
    words_idf = {}
    length = {}

    for word in query_words:
      tf_values, idf_value, length = binary_search_term(self.block_filenames[0], word)
      if tf_values:
        words_tf[word] = tf_values
        words_idf[word] = idf_value

    query_tf = self.calculate_tf(query_words)
    query_tfidf = {}

    for word, tf in query_tf.items():
      idf = words_idf.get(word, 0)
      query_tfidf[word] = tf * idf

    scores = {}

    for word, tfidf in query_tfidf.items():
      if word in words_tf:
        for doc_id, tf_value in words_tf[word]:
          if doc_id not in scores:
            scores[doc_id] = 0
          scores[doc_id] += tfidf * tf_value

    for doc_id in list(scores.keys()):
      scores[doc_id] /= length[doc_id]

    result = sorted(scores.items(), key=lambda tup: tup[1], reverse=True)

    return result[:k]
  
  def get_ordered_block_filenames(self, final_dir):
    block_filenames = [os.path.join(final_dir, f) for f in os.listdir(final_dir) if f.startswith('block_')]
    block_filenames.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
    return block_filenames
  
  def load_index(self):
    final_dir = os.path.join(self.index_dir, 'final')
    self.block_filenames = [self.get_ordered_block_filenames(final_dir)]
    print("Index loaded")

if __name__ == '__main__':
  bsbi = BSBI(100, 1000, './Data/spotify_songs.csv', 'spanish', 'es', 'bsbi_index')
  # bsbi.build_index()
  # print("Index built successfully")

  bsbi.load_index()
  query = "Despacito Quiero respirar tu cuello despacito"
  result = bsbi.retrieval(query)

  print("Query results:")
  for doc_id, score in result:
    print(f"Doc ID: {doc_id}, Score: {score}")

  print("Query completed")
