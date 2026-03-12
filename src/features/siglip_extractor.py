from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


def build_siglip_text(article: dict[str, str]) -> str:
    title = article.get("title", "").strip()
    abstract = article.get("abstract", "").strip()
    if abstract:
        return f"{title} [SEP] {abstract}"
    return title


def load_news_image(image_dir: Path, news_id: str, image_size: int = 224) -> Image.Image:
    image_path = image_dir / f"{news_id}.jpg"
    if image_path.exists():
        return Image.open(image_path).convert("RGB")
    return Image.new("RGB", (image_size, image_size), color=(0, 0, 0))


class SigLIPFeatureExtractor:
    def __init__(self, model_name: str, device: str, max_length: int = 64) -> None:
        self.device = device
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode_batch(self, texts: list[str], images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        image_inputs = self.processor(images=images, return_tensors="pt")

        text_inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}
        image_inputs = {name: tensor.to(self.device) for name, tensor in image_inputs.items()}

        text_embeddings = self.model.get_text_features(**text_inputs)
        image_embeddings = self.model.get_image_features(**image_inputs)
        return text_embeddings.cpu(), image_embeddings.cpu()