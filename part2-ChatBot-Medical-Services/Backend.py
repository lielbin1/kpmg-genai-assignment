from fastapi import FastAPI
from pydantic import BaseModel
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load base prompts
with open("content/prompts/prompt_user_data.txt", encoding="utf-8") as f:
    BASE_COLLECT_INFO_PROMPT = f.read()

with open("content/prompts/prompt_extract_data.txt", encoding="utf-8") as f:
    EXTRACT_DATA_PROMPT = f.read()

with open("content/prompts/prompt_qa.txt", encoding="utf-8") as f:
    QA_PROMPT_TEMPLATE = f.read()

# Return language-specific version of the user info collection prompt
def get_user_info_prompt(language: str) -> str:
    if language.lower() == "hebrew":
        return BASE_COLLECT_INFO_PROMPT + "\n\nענה תמיד בעברית בלבד לאורך כל השיחה."
    return BASE_COLLECT_INFO_PROMPT + "\n\nYou must always respond in English throughout the conversation."

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")


# Basic language detection based on Hebrew characters
def detect_language(text: str) -> str:
    return "Hebrew" if any(c in text for c in "אבגדהוזחטיכלמנסעפצקרשת") else "English"

# Define the schema for incoming chat requests
class ChatRequest(BaseModel):
    user_message: str
    history: list[tuple[str, str]]  # (user, bot)
    user_info: dict
    language: str


# Get embedding using Azure OpenAI embedding model
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise

# Load precomputed embeddings from disk
def load_all_embeddings(folder):
    all_chunks = []
    if not os.path.exists(folder):
        print(f"Warning: Embeddings folder '{folder}' not found")
        return all_chunks
    
    try:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, encoding="utf-8") as f:
                            data = json.load(f)
                            all_chunks.extend(data)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
    except Exception as e:
        print(f"Error walking directory {folder}: {e}")
    
    return all_chunks

# Find top-k most relevant chunks based on cosine similarity
def find_top_k_matches(question, all_chunks, k=5):
    question_vec = np.array(get_embedding(question)).reshape(1, -1)
    
    seen = set()
    scores = []

    for chunk in all_chunks:
        key = (chunk["text"], chunk["source"])
        if key in seen:
            continue
        seen.add(key)

        try:
            doc_vec = np.array(chunk["embedding"]).reshape(1, -1)
            score = cosine_similarity(question_vec, doc_vec)[0][0]
            scores.append((score, chunk["text"], chunk["source"]))
        except Exception as e:
            logger.warning(f"Error computing similarity for chunk {key}: {e}")
            continue

    top = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
    return top

# Load all static embeddings once at server start
ALL_EMBEDDINGS = load_all_embeddings("content/phase2_embedding")

# Translate English text to Hebrew if needed
def translate_to_hebrew(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a translation assistant that translates text from English to Hebrew. Return only the translated sentence, without explanations."},
                {"role": "user", "content": f"Translate the following to Hebrew:\n{text}"}
            ],
            temperature=0.2
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text  

# Main endpoint for handling chat requests
@app.post("/ask")
def ask(request: ChatRequest):
    logger.info(f"📥 Incoming user_info: {request.user_info}")
    try:
        # Define fields to be extracted
        expected_fields = [
            "first_name", "last_name", "id_number", "gender",
            "age", "hmo", "card_number", "membership_tier"
        ]

        # Detect conversation language
        message_language = detect_language(request.user_message)
        conversation_language = request.language or message_language
        logger.info(f"Detected/requested language: {conversation_language}")

        # Detect conversation language
        last_bot_reply = request.history[-1][1] if request.history else ""
        expected_confirmation_responses = [
            "מצוין, אני מזהה אותך עכשיו. אפשר להמשיך! מה תרצה לדעת?",
            "Great! I now recognize you. Feel free to ask your question."
        ]
        is_confirmed = last_bot_reply.strip() in expected_confirmation_responses

        # Extract user info from full chat history
        if is_confirmed and not request.user_info.get("confirmed"):
            logger.info("🟢 GPT confirmation detected. Extracting user data from chat history...")

            extract_prompt = EXTRACT_DATA_PROMPT + f"\n\nConversation language: {conversation_language}\n\nChat log:\n"
            for user_msg, bot_reply in request.history:
                extract_prompt += f"User: {user_msg}\nAssistant: {bot_reply}\n"
            extract_prompt += f"User: {request.user_message}"

            extraction = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": extract_prompt}
                ],
                temperature=0
            )
            try:
                logger.warning(f"📝 Raw GPT extraction response:\n{extraction.choices[0].message.content}")
                
                raw_response = extraction.choices[0].message.content.strip()
                start = raw_response.find("{")
                end = raw_response.rfind("}") + 1
                json_content = raw_response[start:end]

                extracted = json.loads(json_content)
                for key in expected_fields:
                    request.user_info[key] = extracted.get(key)
                request.user_info["confirmed"] = True

                logger.info("📋 Extracted user info from GPT:")
                for key, value in extracted.items():
                    logger.info(f"  {key}: {value}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to parse extracted JSON: {e}")

        # If user is confirmed, continue to answer questions using RAG
        if request.user_info.get("confirmed"):
            logger.info(f"💬 Confirmed: {request.user_info.get('confirmed')}, HMO: {request.user_info.get('hmo')}, Tier: {request.user_info.get('membership_tier')}")
            # Translate question to Hebrew for embedding matching
            if message_language.lower() != "hebrew":
                translated_query = translate_to_hebrew(request.user_message)
                translated_hmo = translate_to_hebrew(request.user_info.get("hmo", ""))
                translated_tier = translate_to_hebrew(request.user_info.get("membership_tier", ""))
                logger.info(f"Translated query: {translated_query} | HMO: {translated_hmo} | Tier: {translated_tier}")
            else:
                translated_query = request.user_message
                translated_hmo = request.user_info.get("hmo", "")
                translated_tier = request.user_info.get("membership_tier", "")


            full_query = f"{translated_query} {translated_hmo} {translated_tier}"
            top_chunks = find_top_k_matches(full_query, ALL_EMBEDDINGS, k=10)


            context_texts = "\n".join(
                f"- {text.strip().replace(chr(10), ' ')}" for score, text, source in top_chunks
            )

            # log
            logger.info("📄 Top relevant chunks:")
            for score, text, source in top_chunks:
                logger.info(f"🔢 Score: {score:.4f} | 📄 Source: {source} | 📝 Text: {text}")

            rag_prompt = QA_PROMPT_TEMPLATE.format(
                **request.user_info,
                conversation_language=conversation_language
            )
            chat_history = "\n".join(
                f"User: {u}\nAssistant: {b}" for u, b in request.history[-4:]
            )

            full_prompt = f"""{rag_prompt}
            User Profile:
            {json.dumps(request.user_info, ensure_ascii=False, indent=2)}

            Recent Conversation:
            {chat_history}

            Context:
            {context_texts}

            Question:
            {request.user_message}
            """

            
            rag_response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides personalized answers based on user profile and context."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.4
            )

            answer = rag_response.choices[0].message.content

            return {
                "response": answer,
                "all_user_info_collected": True,
                "language": conversation_language,
                "user_info": request.user_info 
            }
        # If not yet confirmed, continue collecting missing fields
        all_filled = all(request.user_info.get(field) for field in expected_fields)
        user_prompt = get_user_info_prompt(conversation_language)
        messages = [{"role": "system", "content": user_prompt}]
        for user_msg, bot_reply in request.history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_reply})
        messages.append({"role": "user", "content": request.user_message})

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.6
        )
        reply = response.choices[0].message.content
        return {
            "response": reply,
            "all_user_info_collected": all_filled,
            "language": conversation_language,
            "user_info": request.user_info
        }
    
    # Catch any unexpected error in the whole process
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "response": f"שגיאה: {str(e)}",
            "all_user_info_collected": False,
            "language": request.language or "unknown",
            "user_info": request.user_info
        }