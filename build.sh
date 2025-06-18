#!/bin/bash

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Attempting to download ChromaDB from Hugging Face..."
mkdir -p db # Ensure the 'db' directory exists

# IMPORTANT: Replace 'your-username/hindu-scripture-db' with YOUR actual Hugging Face dataset repo_id
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='your-username/hindu-scripture-db', repo_type='dataset', local_dir='db', resume_download=True)"

if [ $? -ne 0 ]; then
    echo "Error downloading ChromaDB. Exiting."
    exit 1
fi

echo "ChromaDB download complete."
