import os
import logging
from dotenv import load_dotenv
from langdetect import detect
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# load_dotenv()
# AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
# AZURE_KEY = os.getenv("AZURE_KEY")
# FILE_PATH = "phase1_data/283_ex2.pdf"  


def detect_language_langdetect(text: str) -> str:
    try:
        lang = detect(text)
        logger.info(f"Detected language using langdetect: {lang}")
        return lang
    except Exception as e:
        logger.warning(f"Language detection failed (langdetect): {e}")
        return "en"



def get_sample_text_from_result(result, max_words=50):
    all_lines = [line.content for page in result.pages for line in page.lines]
    words = " ".join(all_lines).split()
    return " ".join(words[:max_words])


def analyze_document(file_path: str, endpoint: str, key: str):
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", document=f)
        initial_result = poller.result()

    sample_text = get_sample_text_from_result(initial_result)
    language = detect_language_langdetect(sample_text)

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=f,
            locale=language
        )
        final_result = poller.result()

    return final_result, language



def print_extracted_text(result, language):
    for page_num, page in enumerate(result.pages, start=1):
        print(f"\n---- Page {page_num} ----")
        for line in page.lines:
            print(line.content)
    print("language: ",language)


# if __name__ == "__main__":
#     if not os.path.exists(FILE_PATH):
#         print(f"File not found: {FILE_PATH}")
#     else:
#         result, language = analyze_document(
#             file_path=FILE_PATH,
#             endpoint=AZURE_ENDPOINT,
#             key=AZURE_KEY
#         )
#         print_extracted_text(result, language)
