#!/usr/bin/env python3
import os
import time
import hashlib
import zlib
import gmpy2
from py_ecc.optimized_bls12_381 import curve_order, G1, G2, Z1, Z2, multiply, normalize, pairing
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
import requests
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from requests.exceptions import RequestException
import psutil

# Set the correct directory for caching
cache_dir = '/home/tt/hugginface/hub'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 (with 4GB memory)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_per_process_memory_fraction(0.875)  # Limit GPU memory usage to 87.5% (around 3.5 GB)

print("Using GPU:", torch.cuda.get_device_name(device))

# Download and process books from Project Gutenberg
def download_books(book_ids):
    books = []
    for book_id in book_ids:
        local_file = f"book_{book_id}.txt"
        if os.path.exists(local_file):
            print(f"Book {book_id} already downloaded, loading from file")
            with open(local_file, 'r', encoding='utf-8') as f:
                books.append(f.read())
        else:
            try:
                print(f"Downloading book ID {book_id}")
                text = strip_headers(load_etext(book_id)).strip()
                books.append(text)
                # Save the book locally
                with open(local_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                time.sleep(5)  # Wait 5 seconds between downloads
            except Exception as e:
                print(f"Error downloading book ID {book_id}: {e}")
    return books

def process_books(books, block_size):
    blocks = []
    for book in books:
        for i in range(0, len(book), block_size):
            block = book[i:i+block_size].encode('utf-8')
            blocks.append(block)
    return blocks

# Encode each block using the five encoding methods
def encode_block(block):
    prng_hash = str(gmpy2.mpz_random(gmpy2.random_state(), 2**256))
    crc_hash = zlib.crc32(block)
    checksum = sum(block)
    sha256_hash = hashlib.sha256(block).hexdigest()
    
    # Use py_ecc for error correction coding
    private_key = 42  # Replace with a random scalar value
    public_key = normalize(multiply(G1, private_key))
    ecc_encoded = (public_key, block)
    
    return prng_hash, crc_hash, checksum, sha256_hash, ecc_encoded

# Create linked lists for each encoded block
class DataBlock:
    def __init__(self, data, next_block=None):
        self.data = data
        self.next_block = next_block

def create_linked_list(encoded_block):
    head = DataBlock(encoded_block)
    return head

# Train the neural networks
class EncodedDataset(torch.utils.data.IterableDataset):
    def __init__(self, current_blocks, next_blocks, max_length):
        self.current_blocks = current_blocks
        self.next_blocks = next_blocks
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token
        self.max_length = max_length

    def __iter__(self):
        for idx in range(len(self.current_blocks)):
            current_block = ' '.join(map(str, self.current_blocks[idx]))
            next_block = ' '.join(map(str, self.next_blocks[idx]))
            current_inputs = self.tokenizer(current_block, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
            next_inputs = self.tokenizer(next_block, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
            yield current_inputs.input_ids.squeeze(), current_inputs.attention_mask.squeeze(), next_inputs.input_ids.squeeze()

def train_model(current_blocks, next_blocks, model, batch_size, max_length, accumulation_steps=8):
    dataset = EncodedDataset(current_blocks, next_blocks, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    torch.cuda.set_per_process_memory_fraction(0.875, device=device)  # Limit maximum GPU memory allocation to 87.5% (around 3.5 GB)

    for epoch in range(3):
        for batch_idx, (current_ids, current_mask, next_ids) in enumerate(dataloader):
            current_ids = current_ids.to(device)
            current_mask = current_mask.to(device)
            next_ids = next_ids.to(device)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            outputs = model(input_ids=current_ids, attention_mask=current_mask, labels=next_ids)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}")
        
        cpu_memory_usage = psutil.Process().memory_info().rss / 1024 ** 3  # Get CPU memory usage in GB
        gpu_memory_usage = torch.cuda.memory_allocated() / 1024 ** 3  # Get GPU memory usage in GB
        print(f"CPU Memory Usage: {cpu_memory_usage:.2f} GB")
        print(f"GPU Memory Usage: {gpu_memory_usage:.2f} GB")

def save_model(model, model_name):
    model_path = f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

def load_model(model, model_name):
    model_path = f"{model_name}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded: {model_path}")
    else:
        print(f"Model not found: {model_path}")

def download_and_train_models(current_blocks, next_blocks, batch_size, max_length):
    model_names = ['gpt2', 'gpt2-medium', 'distilgpt2']
    trained_models = []
    
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        local_model_path = f"{model_name}_model"
        if os.path.exists(local_model_path):
            print(f"Loading local model: {model_name}")
            model = GPT2LMHeadModel.from_pretrained(local_model_path, cache_dir=cache_dir).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(local_model_path, cache_dir=cache_dir)
        else:
            print(f"Downloading model: {model_name}")
            for attempt in range(5):  # Try up to 5 times
                try:
                    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
                    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                    # Save the model locally for future use
                    model.save_pretrained(local_model_path)
                    tokenizer.save_pretrained(local_model_path)
                    print(f"Model {model_name} saved locally")
                    break
                except RequestException as e:
                    print(f"Download failed (attempt {attempt + 1}/5): {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {model_name} after 5 attempts")
                continue
        
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        model.config.pad_token_id = model.config.eos_token_id  # Update model config
        
        print(f"Training model: {model_name}")
        train_model(current_blocks, next_blocks, model, batch_size, max_length)
        save_model(model, model_name)
        
        model.tokenizer = tokenizer  # Attach tokenizer to the model for easier access
        trained_models.append(model)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    return trained_models

# Retrieve the sequence of blocks
def predict_next_block(current_block, model):
    inputs = model.tokenizer(str(current_block), return_tensors='pt').to(device)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    outputs = model.generate(inputs.input_ids, max_length=256)
    next_block = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return next_block

def combine_blocks(blocks):
    return b''.join(blocks)

# Main script
if __name__ == '__main__':
    book_ids = [2600, 1399, 84, 11, 1342, 1661, 76, 98, 1260, 844, 1080, 16328, 174, 4300, 345, 1497, 30254, 2701, 2814, 140]
    block_size = 1024
    batch_size = 1  # Reduced batch size to 1
    max_length = 256  # Reduced max length to 256
    accumulation_steps = 8  # Accumulate gradients over 8 batches

    print("Downloading books...")
    books = download_books(book_ids)

    print("Processing books into blocks...")
    blocks = process_books(books, block_size)

    print("Encoding blocks...")
    encoded_blocks = [encode_block(block) for block in blocks]

    print("Creating linked lists...")
    linked_lists = [create_linked_list(encoded_block) for encoded_block in encoded_blocks]

    print("Downloading and training models...")
    current_blocks = [linked_list.data for linked_list in linked_lists[:-1]]
    next_blocks = [linked_list.data for linked_list in linked_lists[1:]]
    models = download_and_train_models(current_blocks, next_blocks, batch_size, max_length)

    print("Retrieving sequence of blocks...")
    for model, model_name in zip(models, ['gpt2', 'gpt2-medium', 'distilgpt2']):
        print(f"Using model: {model_name}")
        reconstructed_blocks = []
        current_block = linked_lists[0].data
        for _ in range(len(linked_lists) - 1):
            next_block = predict_next_block(current_block, model)
            reconstructed_blocks.append(next_block)
            current_block = next_block

        print("Combining reconstructed blocks...")
        reconstructed_data = combine_blocks(reconstructed_blocks)

        print(f"Results for {model_name}:")
        print("Original data size:", len(combine_blocks(blocks)))
        print("Reconstructed data size:", len(reconstructed_data))
        print("Data match:", reconstructed_data == combine_blocks(blocks))
        print("\n")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
