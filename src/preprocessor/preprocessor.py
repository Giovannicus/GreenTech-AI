import requests
import shutil
import os
import random
from collections import defaultdict
import os
import requests
import tarfile

def download_and_extract(url, extract_to="."):
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

    print("Removing unnecessary files...")
    # Lista di pattern di file da rimuovere
    unwanted_patterns = [
        '.DS_Store',      # File di sistema Mac
        '._*',            # File di risorse Mac
        '.AppleDouble',   # File di sistema Mac
        '__MACOSX',      # Cartella di sistema Mac
        'Thumbs.db',      # File di anteprima Windows
        '.directory'      # File KDE
    ]

    # Rimuove i file indesiderati
    for root, dirs, files in os.walk("progetto-finale-flowes", topdown=True):
        for pattern in unwanted_patterns:
            # Rimuove file che corrispondono al pattern
            for name in files:
                if name == pattern or (pattern.startswith('._') and name.startswith('._')):
                    file_path = os.path.join(root, name)
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
            
            # Rimuove cartelle che corrispondono al pattern
            for name in dirs[:]:  # Usa una copia della lista per modificarla durante l'iterazione
                if name == pattern:
                    dir_path = os.path.join(root, name)
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                    dirs.remove(name)

    # Rimuove il file tar.gz originale
    os.remove(tar_path)
    os.remove("._progetto-finale-flowes")
    print(f"Files extracted and cleaned in: {extract_to}")
    os.rename("progetto-finale-flowes", "dataset")

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
