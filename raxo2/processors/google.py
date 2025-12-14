import os
import json
import itertools
import random
from google import genai
from google_images_search import GoogleImagesSearch


class GoogleProcessor:
    """
    Processor per la generazione di query Google Image Search.
    
    Supporta due modalità:
    - direct: L'LLM genera direttamente le query complete
    - compositional: L'LLM genera liste di attributi, poi combinati programmaticamente
    """
    
    def __init__(self, google_api_key: str, google_cx: str, gemini_api_key: str):
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = "gemini-2.5-flash" 

        # Prompt per generazione DIRETTA (LLM genera query complete)
        self.prompt_direct = '''
Generate {N_QUERIES} Google Images search queries for: "{OBJECT_CLASS}"
PURPOSE: airport X-ray scanner shape learning

OBJECT CONSTRAINTS:
- Must fit in luggage (≤120 cm)
- Exclude industrial machinery, vehicle parts, fixed installations

QUERY DIVERSITY RULES:
- Cover visually distinct variants relevant to X-ray appearance
  (commercial, modified, improvised when applicable)
- Include different configurations or states ONLY if meaningful for the object
- Use multiple viewpoints when they affect silhouette or internal layout

FORMAT:
- 3-8 words per query
- Comma-separated list only
- No explanations

Example (knife):
kitchen knife top view, folding knife open, ceramic blade flat lay, improvised metal blade
'''


        # Prompt per generazione COMPOSIZIONALE (LLM genera liste di attributi)
        self.prompt_compositional = '''
Generate Google search query attributes for "{OBJECT_CLASS}" 
(airport X-ray image classification research).

Select 3-4 MUTUALLY EXCLUSIVE attribute categories that are visually and structurally
relevant for identifying this object in X-ray scans.

Guidelines:
- Categories must be appropriate for the object class (do NOT force generic ones).
- Each category: 3-6 concrete, non-overlapping values.
- Objects must fit in luggage (≤120 cm). Exclude industrial or fixed installations.
- Prefer attributes that change X-ray appearance (shape, density, configuration).

Commonly useful categories (use only if relevant):
- VARIANTS (commercial, modified, improvised)
- CONFIGURATION / STATE (assembled, folded, open, loaded, etc.)
- MATERIAL / DENSITY (if affects X-ray signature)
- VIEWPOINTS (top, side, angled, flat lay)

OUTPUT:
JSON array of arrays only. No explanations, no labels.

Example:
[
  ["kitchen knife", "box cutter", "ceramic blade", "improvised blade"],
  ["open", "closed", "folded"],
  ["top view", "side profile", "angled view"]
]
'''

    def generate_queries_direct(self, object_class: str, n_queries: int = 10) -> list[str]:
        """
        Genera query usando l'approccio diretto: LLM restituisce query complete.
        
        Args:
            object_class: Nome della classe oggetto
            n_queries: Numero di query da generare
            
        Returns:
            Lista di query di ricerca
        """
        formatted_prompt = self.prompt_direct.replace("{OBJECT_CLASS}", object_class)
        formatted_prompt = formatted_prompt.replace("{N_QUERIES}", str(n_queries))
        print(f"[Direct] Generating {n_queries} queries for category: {object_class}...")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=formatted_prompt
            )
            queries_str = response.text
            queries = [q.strip() for q in queries_str.split(',') if q.strip()]
            print(f"[Direct] Generated {len(queries)} queries: {queries}")
            return queries
        except Exception as e:
            print(f"Error generating queries with LLM: {e}")
            raise e

    def generate_queries_compositional(
        self, 
        object_class: str, 
        n_queries: int = 10
    ) -> list[str]:
        """
        Genera query usando l'approccio composizionale:
        1. LLM genera liste di attributi mutuamente esclusivi
        2. Combinazione programmatica tramite sampling casuale senza duplicati
        
        Args:
            object_class: Nome della classe oggetto
            n_queries: Numero di query da generare
            
        Returns:
            Lista di query di ricerca composte
        """
        formatted_prompt = self.prompt_compositional.replace("{OBJECT_CLASS}", object_class)
        print(f"[Compositional] Generating attribute lists for category: {object_class}...")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=formatted_prompt
            )
            response_text = response.text.strip()
            
            # Rimuove eventuale markdown wrapper
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            
            attribute_lists = json.loads(response_text)
            print(f"[Compositional] LLM returned {len(attribute_lists)} attribute categories:")
            for i, cat in enumerate(attribute_lists):
                print(f"  Category {i+1}: {cat}")
            
            # Genera tutte le combinazioni possibili (prodotto cartesiano)
            all_combinations = list(itertools.product(*attribute_lists))
            print(f"[Compositional] Total possible combinations: {len(all_combinations)}")
            
            # Sampling casuale senza duplicati
            n_to_sample = min(n_queries, len(all_combinations))
            sampled_combinations = random.sample(all_combinations, n_to_sample)
            
            # Componi le query unendo gli elementi di ogni combinazione
            queries = [" ".join(combo) for combo in sampled_combinations]
            print(f"[Compositional] Generated {len(queries)} queries: {queries}")
            
            return queries
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Response was: {response.text}")
            # Fallback al metodo diretto
            print("Falling back to direct query generation...")
            return self.generate_queries_direct(object_class)
        except Exception as e:
            print(f"Error generating compositional queries: {e}")
            raise e

    def generate_queries(
        self, 
        object_class: str, 
        mode: str = "direct",
        n_queries: int = 10
    ) -> list[str]:
        """
        Factory method per generare query nella modalità specificata.
        
        Args:
            object_class: Nome della classe oggetto
            mode: "direct" o "compositional"
            n_queries: Numero di query (usato solo per compositional)
            
        Returns:
            Lista di query di ricerca
        """
        if mode == "compositional":
            return self.generate_queries_compositional(object_class, n_queries)
        return self.generate_queries_direct(object_class, n_queries)

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