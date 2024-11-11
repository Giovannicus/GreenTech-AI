import requests
import zipfile
import os
import random
from collections import defaultdict
import os
import requests
import tarfile

def download_and_extract(url, extract_to="."):
    # Crea la directory se non esiste
    os.makedirs(extract_to, exist_ok=True)
    
    # Percorso dove salvare il file .tar.gz
    tar_path = os.path.join(extract_to, 'dataset.tar.gz')
    
    # Download del file
    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    with open(tar_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    
    # Estrazione del file
    print("Extracting files...")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
    
    # Rimozione del file .tar.gz
    os.remove(tar_path)
    print(f"Files extracted in: {extract_to}")

def get_balanced_indices(dataset, num_samples):
    
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset.imgs[idx]
        class_indices[label].append(idx)

    num_classes = len(class_indices)
    samples_per_class = num_samples // num_classes

    balanced_indices = []
    for class_idx in class_indices:
        indices = class_indices[class_idx]
        random.shuffle(indices)
        balanced_indices.extend(indices[:samples_per_class])

    random.shuffle(balanced_indices)
    return balanced_indices
