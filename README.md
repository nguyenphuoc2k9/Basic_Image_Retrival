
1. Clone the repository:
```bash

git  clone  https://github.com/nguyenphuoc2k9/Email_classification

cd  Email_classification

```

  

2. (Optional) Create and activate a virtual environment:

  

- For Unix/macOS:

  

```bash

python3  -m  venv  .venv

source  .venv/bin/activate

```

  

- For Windows:

  

```bash

python  -m  venv  venv

.\venv\Scripts\activate

```

  

3. Install the required dependencies:

  

```bash

pip  install  -r  requirements.txt

```

  4. import required module
  ```
import os
import chromadb
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
```
5. Get data
```
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
HNSW_SPACE = "hnsw:space"
```
```
def  get_files_path(path):
	files_path = []
	for label in CLASS_NAME:
	label_path = path + "/" + label
	filenames = os.listdir(label_path)
	for filename in filenames:
	filepath = label_path + '/' + filename
	files_path.append(filepath)
	return files_path
```
```
data_path = f'{ROOT}/train'
files_path = get_files_path(path=data_path)
```
6. Image embedding
```
embedding_function = OpenCLIPEmbeddingFunction()
def  get_single_image_embedding(image):
	embedding = embedding_function._encode_image(image=np.array(image))
	return embedding
```
```
img = Image.open('data/train/African_crocodile/n01697457_260.JPEG')
get_single_image_embedding(image=img)
```
7. Chromadb Embedding Collection
```
def  add_embedding(collection, files_path):
	ids = []
	embeddings = []
	for id_filepath, filepath in tqdm(enumerate(files_path)):
		ids.append(f'id_{id_filepath}')
		image = Image.open(filepath)
		embedding = get_single_image_embedding(image=image)
		embeddings.append(embedding)
		collection.add(
			embeddings=embeddings,
			ids=ids
		)
```
```
# Create a Chroma Client
chroma_client = chromadb.Client()

# Create L2 collection
l2_collection = chroma_client.get_or_create_collection(name="l2_collection",
metadata={HNSW_SPACE:  "l2"})

#Create cosine similarity collection
cosine_collection = chroma_client.get_or_create_collection(name='l2_collection',
metadata={HNSW_SPACE: "cosine"})

add_embedding(collection=<l2_collection || cosine_collection>, files_path=files_path)
```
8. Search Image
```
def  search(image_path, collection, n_results):
	query_image = Image.open(image_path)
	query_embedding = get_single_image_embedding(query_image)
	results = collection.query(
		query_embeddings=[query_embedding],
		n_results = n_results # how many results to return
	)
	return results
```
```
test_path = f'{ROOT}/test'
test_files_path = get_files_path(path=test_path)
test_path = test_files_path[1]
#l2 search
l2_results = search(image_path=test_path, collection=l2_collection, n_results=5)

#cosine search
cosine_results = search(image_path=test_path,collection=cosine_collection,n_results=5)
```
