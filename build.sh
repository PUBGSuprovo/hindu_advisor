#!/bin/bash

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Attempting to download ChromaDB from Hugging Face..."
mkdir -p db # Ensure the 'db' directory exists

# IMPORTANT:  The repo_id is now CORRECTED to: Dyno1307/hindu_db
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dyno1307/hindu_db', repo_type='dataset', local_dir='db', resume_download=True)"

if [ $? -ne 0 ]; then
    echo "Error downloading ChromaDB. Exiting."
    exit 1
fi

echo "ChromaDB download complete."
