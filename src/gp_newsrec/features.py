from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import ensure_dir, read_jsonl


def load_prepared_news(data_dir: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(Path(data_dir) / "news.jsonl"))


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _news_text(news: dict[str, Any]) -> str:
    parts = [
        str(news.get("category", "")),
        str(news.get("subcategory", "")),
        str(news.get("title", "")),
        str(news.get("abstract", "")),
    ]
    return " [SEP] ".join(part for part in parts if part)


def extract_siglip_features(
    data_dir: str | Path = "artifacts/data",
    image_dir: str | Path = "newData",
    output_dir: str | Path = "artifacts/features/siglip",
    model_name: str = "google/siglip-base-patch16-224",
    batch_size: int = 64,
    device: str = "auto",
    max_news: int | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoModel, AutoProcessor

    output_dir = ensure_dir(output_dir)
    image_dir = Path(image_dir)
    news = load_prepared_news(data_dir)
    if max_news is not None:
        news = news[:max_news]

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(selected_device)
    model.eval()

    text_features: list[np.ndarray] = []
    image_features: list[np.ndarray] = []
    news_ids: list[str] = []

    with torch.inference_mode():
        for batch in tqdm(list(_batched(news, batch_size)), desc="SigLIP features"):
            texts = [_news_text(record) for record in batch]
            text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
            text_inputs = {key: value.to(selected_device) for key, value in text_inputs.items()}
            if hasattr(model, "get_text_features"):
                text_emb = model.get_text_features(**text_inputs)
            else:
                text_emb = model(**text_inputs).text_embeds
            text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

            images = []
            for record in batch:
                image_path = Path(record.get("image_path") or image_dir / f"{record['news_id']}.jpg")
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {key: value.to(selected_device) for key, value in image_inputs.items()}
            if hasattr(model, "get_image_features"):
                image_emb = model.get_image_features(**image_inputs)
            else:
                image_emb = model(**image_inputs).image_embeds
            image_emb = torch.nn.functional.normalize(image_emb, dim=-1)

            text_features.append(text_emb.cpu().numpy().astype("float32"))
            image_features.append(image_emb.cpu().numpy().astype("float32"))
            news_ids.extend(str(record["news_id"]) for record in batch)

    text_array = np.concatenate(text_features, axis=0)
    image_array = np.concatenate(image_features, axis=0)
    np.save(output_dir / "text.npy", text_array)
    np.save(output_dir / "image.npy", image_array)
    with (output_dir / "news_ids.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(news_ids, file, ensure_ascii=False)
    with (output_dir / "meta.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            {
                "model_name": model_name,
                "count": len(news_ids),
                "text_dim": int(text_array.shape[1]),
                "image_dim": int(image_array.shape[1]),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "count": len(news_ids),
        "text_dim": int(text_array.shape[1]),
        "image_dim": int(image_array.shape[1]),
        "device": selected_device,
    }
