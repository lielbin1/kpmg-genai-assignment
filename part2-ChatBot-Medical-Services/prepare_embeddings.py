import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = "text-embedding-ada-002"

# Initialize client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_ENDPOINT
)

# Recursively load .txt files from folder
def load_txt_files(folder_path):
    txt_dict = {}
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                full_path = os.path.join(root, file_name)
                with open(full_path, encoding="utf-8") as f:
                    rel_path = os.path.relpath(full_path, folder_path)
                    txt_dict[rel_path] = f.read()
    return txt_dict

# Basic chunking by newline
def prepare_chunks(txt_dict):
    chunks = []
    for filename, content in txt_dict.items():
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                chunks.append({
                    "source": filename,
                    "text": line
                })
    return chunks


# Embed text
def embed_texts(chunks):
    texts = [chunk["text"] for chunk in chunks]
    response = client.embeddings.create(
        model=DEPLOYMENT_NAME,
        input=texts
    )
    for i, embedding in enumerate(response.data):
        chunks[i]["embedding"] = embedding.embedding
    return chunks

def main():
    input_root = "phase2_data_txt"
    output_root = "phase2_embedding"
    os.makedirs(output_root, exist_ok=True)

    txt_data = load_txt_files(input_root)

    for filename, content in txt_data.items():
        chunks = prepare_chunks({filename: content})
        enriched_chunks = embed_texts(chunks)

        output_path = os.path.join(output_root, filename.replace(".txt", ".json"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(enriched_chunks)} embeddings to {output_path}")

if __name__ == "__main__":
    main()
