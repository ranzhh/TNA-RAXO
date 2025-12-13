import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raxo2.processors.google import GoogleProcessor

def main():
    load_dotenv()
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cx = os.getenv("GOOGLE_CX")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([google_api_key, google_cx, gemini_api_key]):
        print("Error: Missing environment variables. Please check your .env file.")
        return

    print("Initializing GoogleProcessor...")
    processor = GoogleProcessor(google_api_key, google_cx, gemini_api_key)
    
    object_class = "wrench"
    print(f"\n--- Testing Query Generation for '{object_class}' ---")
    queries = processor.generate_queries(object_class)
    
    if not queries:
        print("Failed to generate queries.")
        return
        
    print(f"Queens generated: {queries}")

    print(f"\n--- Testing Image Search (downloading 3 images) ---")
    out_dir = "test_downloads"
    
    downloaded_files = processor.search_images(queries[:2], n_images_total=3, out_dir=out_dir)
    
    print(f"\nDownloaded files ({len(downloaded_files)}):")
    for f in downloaded_files:
        print(f" - {f}")

if __name__ == "__main__":
    main()
