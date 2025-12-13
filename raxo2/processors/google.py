import os
from google import genai
from google_images_search import GoogleImagesSearch

class GoogleProcessor:
    def __init__(self, google_api_key: str, google_cx: str, gemini_api_key: str):
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = "gemini-2.5-flash" 

        self.prompt = '''
You are a computer vision research assistant.
Generate Google Image search queries for an object class.
Images are used to extract OBJECT SHAPE only (silhouette / geometry).

Input:
- Object class: "{OBJECT_CLASS}"

Goal:
Maximize intra-class SHAPE variability while keeping images easy to segment.

Instructions:
- Cover different:
  • object sub-types / variants (e.g. machete, cleaver, etc.)
  • geometric states (open/closed, folded/extended, if applicable)
  • viewpoints: top/flat lay, side/profile, front or edge-on, rotated
- Prefer single-object images with minimal background.
  Use keywords like: "isolated", "white background", "transparent background", "flat lay", "PNG".
- Do NOT focus on material, occlusion, or context.
- Avoid illustrations, drawings, icons, or artistic images.
- Keep queries concise (max 10 words) and non-redundant.

Output:
- Output ONLY the queries as a single line
- Separate queries with commas
- Total queries: 15-25
- No explanations, no formatting, no line breaks
'''

    def generate_queries(self, object_class: str) -> list[str]:
        """
        Generates search queries for the given object class using Gemini.
        """
        formatted_prompt = self.prompt.replace("{OBJECT_CLASS}", object_class)
        print(f"Generating queries for category: {object_class}...")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=formatted_prompt
            )
            queries_str = response.text
            queries = [q.strip() for q in queries_str.split(',') if q.strip()]
            print(f"Generated {len(queries)} queries: {queries}")
            return queries
        except Exception as e:
            print(f"Error generating queries with LLM: {e}")
            raise e

    def search_images(self, queries: list[str], n_images_total: int, out_dir: str) -> list[str]:
        """
        Search and download images for the given list of queries.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        downloaded_paths = []
        gis = GoogleImagesSearch(self.google_api_key, self.google_cx)
        
        search_params_base = {
            'num': 10,
            'searchType': 'image',
            'safe': 'off',
            'fileType': 'jpg|png',
            'imgType': 'photo',
            'lr': 'lang_en',
        }

        total_downloaded = 0
        
        for i, query in enumerate(queries):
            if total_downloaded >= n_images_total:
                break
            
            print(f"Processing query {i+1}/{len(queries)}: '{query}'")
            search_params = search_params_base.copy()
            search_params['q'] = query
            
            try:
                gis.search(search_params=search_params)
                for image in gis.results():
                    if total_downloaded >= n_images_total:
                        break
                    
                    try:
                        ext = image.url.split('.')[-1].split('?')[0].lower()
                        if ext not in ['jpg', 'jpeg', 'png']:
                            ext = 'jpg'
                        
                        filename = f"img_{total_downloaded}_{i}.{ext}"
                        full_path = os.path.join(out_dir, filename)
                        
                        image.download(out_dir)
                        if os.path.exists(image.path):
                            os.rename(image.path, full_path)
                            downloaded_paths.append(full_path)
                            total_downloaded += 1
                            print(f"Downloaded: {full_path}")
                        else:
                            print(f"Failed to download: {image.url}")

                    except Exception as e:
                        print(f"Error downloading image {image.url}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error executing query '{query}': {e}")
                continue

        print(f"Finished. Total downloaded: {total_downloaded}")
        return downloaded_paths