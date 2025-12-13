from PIL.Image import Image


class GoogleProcessor:
    def __init__(self, api_key):
        self.api_key = api_key

    def search_images(self, caption: str, n_images: int) -> list[Image]:
        # Placeholder for Google image search logic
        pass
