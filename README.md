#  Proyecto 2 - Base de Datos II
## Introducción

### Objetivo del Proyecto

### Descripción del Dominio de Datos y la Importancia de Aplicar Indexación

## Backend: Índice Invertido

### Construcción del Índice Invertido en Memoria Secundaria

El algoritmo BSBI (Blocked Sort-Based Indexing) permite construir un índice invertido eficientemente utilizando memoria secundaria (disco) en lugar de cargar todo en la RAM. Este método se divide en varias etapas clave: construcción de índices locales, fusión de bloques, y optimización de consultas  textuales mediante similitud de coseno.

#### Componentes Principales

1. **`TextProcessor`**: Clase para preprocesar texto, incluyendo la eliminación de stopwords y la normalización mediante stemming.
2. **`InvertIndex`**: Clase para construir y almacenar un índice invertido.
3. **`BSBI`**: Clase principal que gestiona la creación de bloques de índice, la fusión de estos bloques y la recuperación de información.

#### Optimización
Se implementa una caché LRU (`LRUCache`) para optimizar el acceso a los bloques de índice, reduciendo las operaciones de lectura desde el disco.

#### Descripción de Clases y Métodos

##### Clase `LRUCache`
Esta clase implementa una caché LRU usando `OrderedDict` para manejar el almacenamiento de los bloques de índice más recientemente utilizados.

- **`__init__(self, capacity)`**: Inicializa la caché con una capacidad específica.
- **`get(self, key)`**: Recupera un elemento de la caché y lo mueve al final para marcarlo como recientemente utilizado.
- **`put(self, key, value)`**: Inserta un nuevo elemento en la caché y elimina el menos recientemente utilizado si se supera la capacidad.

##### Clase `TextProcessor`
Esta clase se encarga del preprocesamiento de texto.

- **`__init__(self, lang)`**: Inicializa el procesador de texto con un stemmer y una lista de stopwords para un idioma específico.
- **`preprocess(self, text)`**: Normaliza y elimina stopwords del texto.
- **`calculate_tf(self, doc)`**: Calcula la frecuencia de términos normalizada (tf) de un documento.

##### Clase `InvertIndex`
Esta clase construye y almacena el índice invertido.

- **`__init__(self, index_file, lang)`**: Inicializa el índice invertido con un archivo de destino y un procesador de texto.
- **`calculate_idf(self, collection)`**: Calcula la frecuencia inversa de los documentos (idf) para una colección de documentos.
- **`building(self, data)`**: Construye el índice invertido y almacena los datos en un archivo.

##### Clase `BSBI`
Esta es la clase principal que gestiona la construcción y manejo del índice invertido en bloques.

- **`__init__(self, initial_block_size, block_size, data_path, lang, lang_abbr, index_dir, rebuild=False, cache_size=10)`**: Inicializa los parámetros del índice BSBI y gestiona la construcción o carga del índice.
- **`store_index(self, index_file, index, idf, length)`**: Almacena un bloque de índice en un archivo.
- **`build_local_indexes(self)`**: Construye índices locales a partir de bloques de documentos.
- **`merge_two_sorted_arrays(self, arr1, arr2)`**: Fusiona dos listas ordenadas de tuplas (doc_id, tf).
- **`load_index_block(self, block_filename)`**: Carga un bloque de índice desde el archivo, usando la caché LRU.
- **`merge_two_rows(self, row1, row2, count=0, level=0)`**: Fusiona dos filas de bloques de índice, combinando sus términos.
- **`merge_blocks(self)`**: Gestiona la fusión de todos los bloques de índice hasta que solo queda un bloque final.
- **`clean_dir(self, dir_path)`**: Limpia un directorio especificado.
- **`build_index(self)`**: Construye el índice completo a partir de bloques locales y fusiona los bloques.
- **`retrieval(self, query, k=10)`**: Recupera los documentos más relevantes para una consulta utilizando la búsqueda binaria en los bloques de índice.
- **`get_ordered_block_filenames(self, final_dir)`**: Obtiene una lista ordenada de archivos de bloques en un directorio.
- **`add_index_keys(self, block_filename)`**: Añade las primeras y últimas claves de cada bloque de índice.
- **`load_index(self)`**: Carga el índice desde los archivos de bloques.

#### Optimización de Consultas

- **Caché LRU**: Implementación de una caché LRU para almacenar bloques de índice recientemente utilizados y reducir el tiempo de acceso a disco.
- **Búsqueda Binaria**: Utilización de búsqueda binaria para localizar los bloques relevantes durante las consultas, utilizando una lista de primeras y últimas claves para cada bloque.

#### Flujo del Algoritmo BSBI

1. **Construcción de Índices Locales**: Los documentos se procesan en bloques de tamaño fijo. Cada bloque se indexa y almacena en un archivo separado.
2. **Fusión de Bloques**: Los bloques de índice se fusionan en niveles hasta que solo queda un bloque final.
3. **Consultas**: Las consultas se procesan buscando los términos en los bloques relevantes usando búsqueda binaria, y se calculan los puntajes de relevancia utilizando la similitud del coseno.

##### Construcción de Índices Locales

La construcción del índice invertido comienza con la creación de índices locales para bloques de documentos. Cada bloque se procesa por separado, almacenando su índice en un archivo en el disco.

**Fragmento de código relevante:**

```python
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
        if docs:
            block_index_file = os.path.join(self.index_dir, f'block_{block_num}.pkl')
            invert_index = InvertIndex(block_index_file, self.lang)
            invert_index.building(docs)
            self.block_filenames.append(block_index_file)
    print("Local indexes built")
```

- **Explicación**: Este método procesa el archivo de datos en bloques de tamaño `initial_block_size`. Cada bloque se indexa utilizando la clase `InvertIndex` y se guarda en un archivo (`block_num.pkl`). Los archivos de índice de los bloques se almacenan en el disco, evitando cargar todos los datos en RAM simultáneamente.

##### Clase `InvertIndex` para Construcción de Índices Locales

**Fragmento de código relevante:**

```python
class InvertIndex:
    def __init__(self, index_file, lang):
        self.index_file = index_file
        self.index = {}
        self.idf = {}
        self.length = {}
        self.processor = TextProcessor(lang)

    def building(self, data):
        processed_collection = {doc_id: self.processor.preprocess(text) for doc_id, text in data.items()}
        for doc_id, words in processed_collection.items():
            tf = self.processor.calculate_tf(words)
            for word, tf_value in tf.items():
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append((doc_id, tf_value))
            self.length[doc_id] = np.linalg.norm(list(tf.values()))
        self.calculate_idf(processed_collection)
        self.index = dict(sorted(self.index.items()))
        with open(self.index_file, 'wb') as f:
            pickle.dump((self.index, self.idf, self.length), f)
```

- **Explicación**: La clase `InvertIndex` procesa los documentos en un bloque, calcula las frecuencias de término (tf), y construye el índice invertido para ese bloque. Luego, calcula la frecuencia inversa de documento (idf) y guarda el índice en un archivo utilizando `pickle`.

##### Fusión de Bloques

Una vez que se han construido los índices locales, estos bloques se fusionan en niveles hasta que solo queda un bloque final.

**Fragmento de código relevante:**

```python
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
```

- **Explicación**: Este método fusiona pares de bloques de índice hasta que solo queda un bloque final. Los bloques se combinan nivel por nivel, y los bloques fusionados se almacenan en nuevos archivos. El resultado final es un único bloque de índice almacenado en el directorio `final`.

##### Ejecución Óptima de Consultas aplicando Similitud de Coseno

La recuperación de documentos relevantes se realiza utilizando la similitud de coseno. Esto implica calcular el producto punto entre los vectores tf-idf de la consulta y los documentos, y luego normalizar por la longitud de los vectores.

**Fragmento de código relevante:**

```python
def retrieval(self, query, k=10):
    def binary_search_term(block_keys, blocks, term):
        left, right = 0, len(block_keys) // 2 - 1
        while left <= right:
            mid = (left + right) // 2
            f_key, l_key = block_keys[2 * mid], block_keys[2 * mid + 1]
            if f_key <= term <= l_key:
                index_mid, idf_mid, length = self.load_index_block(blocks[mid])
                if term in index_mid:
                    tf_values = index_mid[term]
                    idf_value = idf_mid.get(term, 0)
                    return tf_values, idf_value, length
                else:
                    return None, None, None
            if term < f_key:
                right = mid - 1
            else:
                left = mid + 1
        return None, None, None

    query_words = self.processor.preprocess(query)
    words_tf = {}
    words_idf = defaultdict(float)
    length = {}

    for word in query_words:
        tf_values, idf_value, local_length = binary_search_term(self.index_keys, self.block_filenames[0], word)
        if tf_values is not None:
            words_tf[word] = tf_values
            words_idf[word] = idf_value
            if not length:
                length = local_length

    query_tf = self.processor.calculate_tf(query_words)
    query_tfidf = {word: tf * words_idf.get(word, 0) for word, tf in query_tf.items()}

    scores = defaultdict(float)
    for word, tfidf in query_tfidf.items():
        if word in words_tf:
            for doc_id, tf_value in words_tf[word]:
                scores[doc_id] += tfidf * tf_value

    for doc_id in list(scores.keys()):
        scores[doc_id] /= length[doc_id]

    result = sorted(scores.items(), key=lambda tup: tup[1], reverse=True)
    return result[:k]
```

- **Explicación**: Este método realiza una consulta utilizando búsqueda binaria para encontrar los términos en los bloques de índice relevantes. Luego, calcula la similitud de coseno entre la consulta y los documentos utilizando los valores tf-idf. Los documentos se ordenan por relevancia y se devuelven los `k` documentos más relevantes.

##### Optimización mediante la Caché LRU

Para mejorar la eficiencia, se utiliza una caché LRU para almacenar bloques de índice recientemente utilizados, minimizando las operaciones de lectura desde el disco.

**Fragmento de código relevante:**

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

- **Explicación**: La caché LRU (Least Recently Used) se implementa usando `OrderedDict`. Almacena un número limitado de bloques de índice y elimina el menos recientemente utilizado cuando se alcanza la capacidad máxima. Esto reduce el número de lecturas desde el disco, mejorando la velocidad de las consultas.

### Ejecución Óptima de Consultas Aplicando Similitud de Coseno

### Construcción del Índice Invertido en PostgreSQL/MongoDB

## Backend: Índice Multidimensional (siguiente entrega)

### Técnica de Indexación de las Librerías Utilizadas (siguiente entrega)

### KNN Search y Range Search (siguiente entrega)

### Análisis de la Maldición de la Dimensionalidad y Cómo Mitigarlo (siguiente entrega)

### API
La API permite realizar búsquedas por letras de canciones usando el índice y obtener información detallada a través de la API de Spotify.

La API está construida usando FastAPI.

### Funcionalidad Principal
- **Búsqueda de Canciones**: La API permite buscar canciones por letra. El índice retorna un arreglo con los top `k` resultados, cada uno conteniendo el ID de la canción y su puntuación de relevancia.
- **Obtención de Información de Spotify**: Utilizando el ID de la canción, la API consulta la API de Spotify para obtener información detallada de la canción, como el nombre, artistas, álbum, URL de previsualización y la imagen del álbum.

#### Endpoint Principal

##### POST `/search`
Permite realizar una búsqueda de canciones.

**Request**
- **Cuerpo de la Solicitud**:
  ```json
  {
    "query": "string",
    "k": 10,
    "language": "spanish",
    "use_postgres": false 
  }
  ```

**Response**
- **Formato de la Respuesta**:
  ```json
  {
    "songs": [
      {
        "id": "string",
        "score": float,
        "name": "string",
        "artists": ["string"],  
        "album": "string", 
        "preview_url": "string",
        "album_image": "string" 
      }
    ],
    "executionTime": float 
  }
  ```

##### Cómo Funciona la API

1. **Inicialización**:
   - La API se inicializa creando dos índices BSBI, uno para canciones en español y otro para canciones en inglés, utilizando los datos de `spotify_songs.csv`.

2. **Recepción de la Solicitud**:
   - El endpoint `/search` recibe una solicitud POST con la consulta de búsqueda, el número de resultados deseados, el idioma de las canciones y si se debe utilizar PostgreSQL o el índice propio para la búsqueda.

3. **Búsqueda en el Índice**:
   - Dependiendo del idioma especificado (`spanish` o `english`), se selecciona el índice BSBI correspondiente.
   - Se realiza la búsqueda en el índice y se obtienen los top `k` resultados (ID de la canción y puntuación).

4. **Consulta a la API de Spotify**:
   - Para cada resultado, se usa el ID de la canción para obtener información detallada desde la API de Spotify.
   - La información obtenida se simplifica para incluir solo los campos necesarios (nombre de la canción, artistas, álbum, URL de previsualización y la imagen del álbum).

5. **Respuesta**:
   - Retorna un objeto JSON con la lista de canciones y el tiempo de ejecución de la búsqueda.

## Frontend
Para el diseño del Frontend optamos por una página web con un diseño minimalista y sencillo para el usuario. Este fue construido usando Next.js y Typescript. El componente principal es la barra de búsqueda, donde se ingresa la letra de la canción. Además, se pueden modificar diferentes parámetros como el Top K resultados, el lenguaje y tipo de índice.

### Diseño de la GUI
El diseño está inspirado en Spotify, ya que las canciones indexadas son de esta plataforma. Lo hicimos simple y usando su paleta de colores.

![Imagen General de la GUI](/readme-images/1.png)

#### Entradas
1. Entrada de la letra de la canción.
![Imagen de la entrada de la letra de la canción](/readme-images/2.png)

2. Entrada del Top k resultados. 
![Imagen de la entrada del Top k resultados](/readme-images/3.png)

3. Entrada del lenguaje del índice.
![Imagen de la entrada del lenguaje del índice](/readme-images/4.png)
![Imagen de la entrada del lenguaje del índice](/readme-images/5.png)

4. Entrada del índice a utilizar.
![Imagen de la entrada del índice a utilizar](/readme-images/6.png)
![Imagen de la entrada del índice a utilizar](/readme-images/7.png)

#### Mini-manual de Usuario
#### Realizar consultas textuales
1. Completar las entradas según sus requerimientos. Y presionar el ícono de buscar.
![Imagen de la entrada del índice a utilizar](/readme-images/8.png)
![Imagen de la entrada del índice a utilizar](/readme-images/9.png)
2. Se retornan los resultados. 
![Imagen de la entrada del índice a utilizar](/readme-images/10.png)

3. Si la canción cuenta con un preview proporcionado por Spotify, podrá pulsar el ícono de reproducir y pausar canción. 
![Imagen de la entrada del índice a utilizar](/readme-images/11.png)
![Imagen de la entrada del índice a utilizar](/readme-images/12.png)

### Análisis Comparativo Visual con Otras Implementaciones
A continuación, las capturas de pantallas de otras plataformas que permiten buscar canciones por su letra:

**Letras**
![Imagen de la entrada del índice a utilizar](/readme-images/13.png)

**Lyrics**
![Imagen de la entrada del índice a utilizar](/readme-images/14.png)

**Find Music By Lyrics**

![Imagen de la entrada del índice a utilizar](/readme-images/15.png)

**Google**
![Imagen de la entrada del índice a utilizar](/readme-images/16.png)

**Spotify**
![Imagen de la entrada del índice a utilizar](/readme-images/17.png)

Nuestro frontend combina lo mejor de otras implementaciones, combinando la limpieza y minimalismo de Google con lo visual de Spotify. Además, diferente a otras soluciones, no sobrecargamos la interfaz con anuncios o paneles, como en Letras y Lyrics. Conservando lo esencial, la barra de búsqueda. 
## Experimentación

### Tablas y Gráficos de los Resultados Experimentales

### Análisis y Discusión
