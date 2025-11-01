import requests
from tqdm import tqdm
import os

def download_dataset(url, path):
    if os.path.exists(path):
        print(f"{path} already exists. Skipping download.")
        return
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(path, 'wb') as file, tqdm(
        desc=path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            bar.update(size)
            
    print(f"Scussfully downloaded {path}")


# train dataset (2.2GB)
url = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt'
path = 'dataset/TinyStoriesV2-GPT4-train.txt'
download_dataset(url, path)

# valid dataset (22.5MB)
url = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'
path = 'dataset/TinyStoriesV2-GPT4-valid.txt'
download_dataset(url, path)