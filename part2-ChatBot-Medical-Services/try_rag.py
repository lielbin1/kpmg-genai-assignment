import os
import json
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load .env variables
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = "text-embedding-ada-002"

# Init Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_ENDPOINT
)

def get_embedding(text):
    response = client.embeddings.create(
        model=DEPLOYMENT_NAME,
        input=[text]
    )
    return response.data[0].embedding

def load_all_embeddings(folder):
    all_chunks = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    all_chunks.extend(data)
    return all_chunks

def find_top_k_matches(question, all_chunks, k=5):
    question_vec = np.array(get_embedding(question)).reshape(1, -1)
    
    seen = set()
    scores = []

    for chunk in all_chunks:
        key = (chunk["text"], chunk["source"]) 
        if key in seen:
            continue
        seen.add(key)

        doc_vec = np.array(chunk["embedding"]).reshape(1, -1)
        score = cosine_similarity(question_vec, doc_vec)[0][0]
        scores.append((score, chunk["text"], chunk["source"]))

    top = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    return top


def main():
    user_question = "כמה הנחה יש לטיפול שורש?"
    
    user_profile = {
        "first_name": "ליאל",
        "last_name": "ביי",
        "id_number": "444444444",
        "gender": "אישה",
        "age": 4,
        "hmo": "מכבי",
        "card_number": "444444444",
        "membership_tier": "זהב"
    }

    # ננסח את השאלה יחד עם הקשר אישי מלא כדי לשפר את ההתאמה
    user_context = (
        f"שם פרטי: {user_profile['first_name']}, "
        f"שם משפחה: {user_profile['last_name']}, "
        f"תעודת זהות: {user_profile['id_number']}, "
        f"מין: {user_profile['gender']}, "
        f"גיל: {user_profile['age']}, "
        f"קופת חולים: {user_profile['hmo']}, "
        f"מספר כרטיס: {user_profile['card_number']}, "
        f"מסלול חברות: {user_profile['membership_tier']}"
    )

    full_query = f"{user_question} {user_profile['hmo']} {user_profile['membership_tier']}"


    all_chunks = load_all_embeddings("phase2_embedding")

    found = [chunk for chunk in all_chunks if "טיפולי שורש / מכבי / זהב" in chunk["text"]]
    for i, chunk in enumerate(found):
        print(f"\n📄 מקור: {chunk['source']}")
        print(f"📝 טקסט: {chunk['text']}")
    

    top_chunks = find_top_k_matches(full_query, all_chunks, k=10)

    print("\n🔍 תוצאות הכי רלוונטיות לשאלה:")
    for score, text, source in top_chunks:
        print(f"\n📄 מקור: {source}")
        print(f"🔢 דמיון: {score:.4f}")
        print(f"📝 טקסט: {text}")


if __name__ == "__main__":
    main()
