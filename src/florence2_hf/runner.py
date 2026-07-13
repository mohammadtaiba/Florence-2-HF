"""Laden und Ausführen von Microsoft Florence-2."""

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoProcessor

from florence2_hf.errors import ImageLoadError, InferenceError, ModelLoadError


LOGGER = logging.getLogger(__name__)


class FlorenceRunner:
    """Kapselt Modellinitialisierung und Bildanalyse."""

    def __init__(self, model_id: str, device_name: str = "auto"):
        self._model_id = model_id
        self._device = self._resolve_device(device_name)
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Lädt Modell und Processor einmalig in den Speicher."""

        LOGGER.info("Lade Modell '%s' auf Gerät '%s'.", self._model_id, self._device)

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=self._dtype,
                trust_remote_code=True,
                use_safetensors=True,
            ).to(self._device)

            self._model.eval()

            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                trust_remote_code=True,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ModelLoadError(
                f"Das Modell '{self._model_id}' konnte nicht geladen werden."
            ) from error

    def run(self, image_path: Path, task_prompt: str, text_input: str | None = None) -> dict[str, Any]:
        """Führt eine Florence-2-Aufgabe für ein lokales Bild aus."""

        if self._model is None or self._processor is None:
            self.load()

        image = self._load_image(image_path)
        prompt = self._build_prompt(task_prompt, text_input)

        try:
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(self._device)
            pixel_values = inputs["pixel_values"].to(
                device=self._device,
                dtype=self._dtype,
            )

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )

            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]

            return self._processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )
        except (RuntimeError, ValueError, KeyError) as error:
            raise InferenceError("Die Florence-2-Ausführung ist fehlgeschlagen.") from error

    @staticmethod
    def _build_prompt(task_prompt: str, text_input: str | None) -> str:
        normalized_text = text_input.strip() if text_input else ""
        return f"{task_prompt}{normalized_text}"

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        if not image_path.is_file():
            raise ImageLoadError(f"Das Bild wurde nicht gefunden: {image_path}")

        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ImageLoadError(
                f"Das Bild konnte nicht gelesen werden: {image_path}"
            ) from error

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        normalized_device = device_name.lower()

        if normalized_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if normalized_device == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError(
                "CUDA wurde angefordert, ist aber in der installierten PyTorch-Version nicht verfügbar."
            )

        if normalized_device not in {"cpu", "cuda"}:
            raise ModelLoadError(
                f"Unbekanntes Gerät '{device_name}'. Erlaubt sind: auto, cpu, cuda."
            )

        return torch.device(normalized_device)
