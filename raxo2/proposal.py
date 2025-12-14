from dataclasses import dataclass
from typing import Callable, Generator

from PIL.Image import Image


@dataclass
class ProposalConfig:
    n_images: int = 10
    mask_threshold: float = 0.5

    image_gather_fn: Callable[[str, int], list[Image]] = None
    mask_extraction_fn: Callable[[Image, str, float], Image] = None
    style_application_fn: Callable[[Image, Image], Image] = None


class ImageProposalMethod:
    def __init__(self, config: ProposalConfig):
        self.config = config

    def _gather_images(self, caption: str) -> list[Image]:
        return self.config.image_gather_fn(caption, self.config.n_images)

    def _extract_masks(self, image: Image, caption: str) -> Image:
        return self.config.mask_extraction_fn(image, caption, self.config.mask_threshold)

    def _apply_style(self, image: Image, mask: Image) -> Image:
        return self.config.style_application_fn(image, mask)

    def run(self, caption: str) -> list[Image]:
        """Given a caption, generate styled images based on extracted masks."""
        images = self._gather_images(caption)
        styled_images = []

        for image in images:
            mask = self._extract_masks(image, caption)
            styled_image = self._apply_style(image, mask)
            styled_images.append(styled_image)

        return styled_images


def google_sam3d_blender_gather_fn(
    caption: str, 
    n_images: int, 
    use_llm: bool = True,
    use_blender: bool = True,
    use_sam3d: bool = True
) -> Generator[Image, None, None]:
    """This is our implementation of a gathering function using Google, SAM/SAM3D, and Blender.

    We first use Google to search for images related to the caption.
    Then, we use SAM3 to extract the mask that is most relevant to the caption.
    We use SAM3D to create a 3D representation of the masked region, export it to Blender.
    Finally, we render the 3D model in Blender to get multiple views of the object.
    
    Args:
        caption: The search query/caption for the object
        n_images: Number of images to gather
        use_llm: If True, use LLM (Gemini) to generate diverse queries. If False, use simple query.
        use_blender: If True, use Blender for multi-view rendering (placeholder)
        use_sam3d: If True, use SAM3D for 3D model generation (placeholder)
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from processors.blender import BlenderProcessor
    from processors.google import GoogleProcessor
    from processors.sam3d import SAMProcessor

    # Initialize processors
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cx = os.getenv("GOOGLE_CX") 
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    google_processor = GoogleProcessor(google_api_key, google_cx, gemini_api_key)
    sam_processor = SAMProcessor()
    blender_processor = BlenderProcessor()

    # Get the images from Google Search
    if use_llm and gemini_api_key:
        queries = google_processor.generate_queries(caption)
        images = google_processor.search_images(queries, n_images, out_dir="temp_images")
    else:
        # Simple query mode
        simple_query = f"A photo of a {caption}"
        images = google_processor.search_images([simple_query], n_images, out_dir="temp_images")

    for image in images:
        if use_sam3d:
            # Extract the mask using SAM3
            obj = sam_processor.model_from_caption(image, caption, threshold=0.5)
            
            if use_blender:
                # Create a 3D model and render it in Blender
                renders = blender_processor.multi_render_obj(obj, n_views=1)
                yield from renders
            else:
                yield obj
        else:
            # Just yield the image if SAM3D is disabled
            yield image

