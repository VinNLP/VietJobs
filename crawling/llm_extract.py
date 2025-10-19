import asyncio
import json
import os
import re
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI

# Percentage of links needed to extract from txt files
REQUIRED_PERCENT = 100

def get_json_link_count(json_file):
    """Get the number of links in a JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

def get_txt_link_count(txt_file):
    """Get the number of non-empty lines in a txt file."""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0

def should_skip_processing(txt_path, json_path):
    """
    Check if processing should be skipped based on link counts.
    Returns True if JSON has >= REQUIRED_PERCENT% of TXT links.
    """
    txt_count = get_txt_link_count(txt_path)
    if txt_count == 0:
        return False
        
    json_count = get_json_link_count(json_path)
    return json_count >= (txt_count * REQUIRED_PERCENT / 100)

# API Keys configuration with rotation support
API_KEYS = [

]

# Global variables for key rotation
current_key_index = 0
chat_model = None

def initialize_chat_model(key_index=0):
    """Initialize the chat model with a specific key index."""
    global chat_model, current_key_index
    current_key_index = key_index
    key_config = API_KEYS[key_index]
    
    chat_model = AzureChatOpenAI(
        api_version=key_config["api_version"],
        azure_endpoint=key_config["azure_endpoint"],
        azure_deployment=key_config["azure_deployment"],
        model=key_config["model"],
        api_key=key_config["api_key"],  # type: ignore
        streaming=True,
        temperature=0.0,
    )
    print(f"Initialized chat model with key index {key_index}")

def rotate_api_key():
    """Rotate to the next available API key."""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    initialize_chat_model(current_key_index)
    print(f"Rotated to API key index {current_key_index}")

def add_api_key(api_version, azure_endpoint, azure_deployment, model, api_key):
    """Add a new API key to the rotation list."""
    API_KEYS.append({
        "api_version": api_version,
        "azure_endpoint": azure_endpoint,
        "azure_deployment": azure_deployment,
        "model": model,
        "api_key": api_key,
    })
    print(f"Added new API key. Total keys: {len(API_KEYS)}")

def get_current_key_info():
    """Get information about the currently active API key."""
    return {
        "index": current_key_index,
        "total_keys": len(API_KEYS),
        "endpoint": API_KEYS[current_key_index]["azure_endpoint"]
    }

# Initialize with the first key
initialize_chat_model(0)


async def crawl_markdown(url):
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        extraction_strategy=None,  # No LLM extraction, just crawl
        cache_mode=CacheMode.BYPASS,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if hasattr(result, "markdown") and result.markdown and hasattr(result.markdown, "raw_markdown"):
            return result.markdown.raw_markdown
        else:
            print(f"Warning: No markdown extracted for {url}")
            return None


def extract_json_from_codeblock(raw_response):
    # Remove code block markers and extract JSON
    match = re.search(r"```json\n(.*?)\n```", raw_response, re.DOTALL)
    if match:
        json_content = match.group(1)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    return None


async def process_url(url, instruction, max_retries=3):
    raw_markdown = await crawl_markdown(url)
    if raw_markdown is None:
        return None

    # Compose the prompt for the LLM
    prompt = f"{instruction}\n\nPage content:\n{raw_markdown}"
    
    # Try with current key, rotate if there's an error
    for attempt in range(max_retries):
        try:
            response = chat_model.invoke(prompt)
            raw_content = response.content

            # Pass the LLM response through extract_json_from_codeblock to get JSON
            json_data = extract_json_from_codeblock(raw_content)
            if json_data is not None:
                # Add the input URL directly to the JSON output instead of relying on AI
                json_data["url"] = url
                return json_data

            # Fallback: return raw LLM response as text with URL
            return {"raw_response": raw_content, "url": url}
            
        except Exception as e:
            print(f"Error with API key {current_key_index} (attempt {attempt + 1}): {e}")
            
            # If this is not the last attempt, try rotating the key
            if attempt < max_retries - 1:
                print("Rotating to next API key...")
                rotate_api_key()
            else:
                print(f"All API keys exhausted for URL: {url}")
                return None
    
    return None


async def process_txt_file(txt_path, instruction, output_dir):
    print(f"Processing {txt_path}...")

    json_filename = Path(txt_path).stem + ".json"
    all_results_file = Path(output_dir) / json_filename

    # Check if we already have enough processed links
    if should_skip_processing(txt_path, all_results_file):
        print(f"Skipping {txt_path} - Required percentage of links already processed")
        return

    # First read all URLs from txt file
    urls = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url:
                urls.append(url)

    total_urls = len(urls)
    required_links = int(total_urls * REQUIRED_PERCENT / 100)
    print(f"Total URLs in file: {total_urls}")
    print(f"Required number of links to process: {required_links}")

    # Convert URLs list to set for efficient lookup
    urls_set = set(urls)

    # Load existing results if any
    all_results = []
    if all_results_file.exists():
        with open(all_results_file, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
                # Filter out results for URLs that are no longer in the txt file
                filtered_results = []
                for result in all_results:
                    url = result.get("url") or result.get("raw_response", {}).get("url")
                    if url in urls_set:
                        filtered_results.append(result)
                    else:
                        print(f"Removing result for URL not in txt file: {url}")
                all_results = filtered_results
                # Write back filtered results
                with open(all_results_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
            except Exception:
                all_results = []

    # Build a set of already processed URLs for skipping
    processed_urls = set()
    for entry in all_results:
        url_val = entry.get("url") or entry.get("raw_response", {}).get("url")
        if url_val:
            processed_urls.add(url_val)

    current_count = len(processed_urls)
    remaining_needed = max(0, required_links - current_count)
    print(f"Already processed: {current_count}")
    print(f"Remaining links needed: {remaining_needed}")

    if remaining_needed == 0:
        print(f"Required number of links already processed for {txt_path}")
        return

    for idx, url in enumerate(urls):
        if current_count >= required_links:
            print(f"Reached required number of links ({required_links})")
            break

        if url in processed_urls:
            continue

        print(f"Processing URL: {url}")
        data = await process_url(url, instruction, max_retries=3)
        if data is not None:
            all_results.append(data)
            current_count += 1
            # Write the updated results after each URL
            with open(all_results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Progress: {current_count}/{required_links} required links")

    print(f"Finished processing {txt_path} -> {all_results_file}")
    print(f"Processed {current_count}/{required_links} required links")


async def main():
    """
    Process a single txt file to extract structured JSON data using LLM and save results to a corresponding JSON file.
    """
    import sys
    if len(sys.argv) != 2:
        print("Usage: python llm_extract.py <txt_file_path>")
        sys.exit(1)

    txt_file = sys.argv[1]
    if not txt_file.endswith('.txt'):
        print("Error: Input file must be a .txt file")
        sys.exit(1)

    if not os.path.exists(txt_file):
        print(f"Error: File {txt_file} does not exist")
        sys.exit(1)

    # Load instruction from file
    try:
        with open("prompts/extraction_prompt.txt", "r") as f:
            instruction = f.read()
    except FileNotFoundError:
        print("Warning: extraction_prompt.txt not found. Using default instruction.")
        instruction = "Extract structured data for job ads according to the schema."

    # Load schema from models/schemas.py
    from models.schemas import JobAdSchema
    schema_json = json.dumps(JobAdSchema.model_json_schema(), ensure_ascii=False, indent=2)

    # Add schema to the instruction for the LLM
    instruction = (
        f"{instruction}\n\n"
        f"Follow this JSON schema for your output:\n"
        f"{schema_json}\n"
        "Return only a valid JSON object."
    )

    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)

    await process_txt_file(txt_file, instruction, output_dir)


if __name__ == "__main__":
    load_dotenv(override=True)
    asyncio.run(main())