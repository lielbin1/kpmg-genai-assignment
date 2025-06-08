import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from extract_fields import analyze_document

load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-05-01-preview"
)


def extract_text_from_document(file_path: str):
    result, language = analyze_document(
        file_path=file_path,
        endpoint=AZURE_ENDPOINT,
        key=AZURE_KEY
    )
    full_text = "\n".join([line.content for page in result.pages for line in page.lines])
    return full_text, language


def load_prompt(language: str, full_text: str):
    file_name = "prompts/prompt_he.txt" if language == "he" else "prompts/prompt_en.txt"
    with open(file_name, encoding="utf-8") as f:
        return f.read() + f'\n\n"""\n{full_text}\n"""'


def call_gpt(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def run_extraction_pipeline(file_path: str):
    full_text, language = extract_text_from_document(file_path)

    # Load the correct base prompt
    file_name = "prompts/prompt_he.txt" if language == "he" else "prompts/prompt_en.txt"
    with open(file_name, encoding="utf-8") as f:
        base_prompt = f.read()
    
    extraction_prompt = base_prompt + f'\n\n"""\n{full_text}\n"""'
    extracted_json = call_gpt(extraction_prompt)

    # Load translation prompt
    with open("prompts/translate_json_fields_prompt.txt", encoding="utf-8") as f:
        translate_prompt = f.read()
    
    translate_full_prompt = translate_prompt + f"\n\n{extracted_json}"
    return_json = call_gpt(translate_full_prompt)

    if return_json.startswith("```json"):
        return_json = return_json.removeprefix("```json").removesuffix("```").strip()
    elif return_json.startswith("```"):
        return_json = return_json.removeprefix("```").removesuffix("```").strip()

    return return_json
