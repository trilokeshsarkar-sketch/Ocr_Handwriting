from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig, TrOCRProcessor, VisionEncoderDecoderModel

try:
    import easyocr
except ImportError:  # pragma: no cover - handled at runtime in the app/CLI
    easyocr = None


@dataclass
class OCRRegion:
    bbox: list[list[float]]
    text: str
    confidence: float
    rect: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]


@dataclass
class OCRLine:
    rect: tuple[int, int, int, int]
    region_count: int


def preprocess_for_detection(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold works well for dense printed pages and scanned documents.
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )


def cleanup_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prefers_easyocr_fallback(primary_text: str, fallback_text: str) -> bool:
    """Choose the fallback when it clearly contains more usable text."""
    primary_text = cleanup_text(primary_text)
    fallback_text = cleanup_text(fallback_text)
    if not fallback_text:
        return False
    if not primary_text:
        return True
    if len(primary_text.split()) <= 2 and len(fallback_text.split()) >= len(primary_text.split()) + 1:
        return True
    return len(fallback_text) >= max(len(primary_text) + 12, int(len(primary_text) * 1.5))


class AdaptiveOCRPipeline:
    def __init__(
        self,
        language_list: list[str] | None = None,
        trocr_model_name: str = "microsoft/trocr-large-handwritten",
        custom_model_path: str | None = None,
    ) -> None:
        if easyocr is None:
            raise ImportError(
                "easyocr is required for AdaptiveOCRPipeline. Install dependencies with "
                "`pip install -r requirements.txt`."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reader = easyocr.Reader(
            language_list or ["en"],
            gpu=torch.cuda.is_available(),
            verbose=False,
        )
        self.trocr_model_name = trocr_model_name
        self.custom_model_path = self.resolve_model_dir(custom_model_path)
        self.processor: TrOCRProcessor | None = None
        self.model: VisionEncoderDecoderModel | None = None
        self.model_source: str | None = None
        self.generation_config = GenerationConfig(
            max_length=96 if self.device == "cpu" else 128,
            num_beams=1 if self.device == "cpu" else 4,
            repetition_penalty=1.05,
            no_repeat_ngram_size=0 if self.device == "cpu" else 2,
        )

    @staticmethod
    def resolve_model_dir(custom_model_path: str | None) -> Path | None:
        if not custom_model_path:
            return None

        candidates: list[Path] = [Path(custom_model_path).expanduser()]

        for candidate in candidates:
            if candidate.is_file():
                continue

            if (candidate / "config.json").exists():
                return candidate

            nested_matches = [path.parent for path in candidate.rglob("config.json")]
            if nested_matches:
                return nested_matches[0]

        return None

    def detect_regions(self, image: Image.Image) -> list[OCRRegion]:
        rgb = np.array(image.convert("RGB"))
        results = self.reader.readtext(rgb, detail=1, paragraph=False)
        if len(results) < 3:
            processed = preprocess_for_detection(image)
            results = self.reader.readtext(processed, detail=1, paragraph=False)
        regions: list[OCRRegion] = []

        for bbox, text, confidence in results:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            rect = (
                int(min(xs)),
                int(min(ys)),
                int(max(xs)),
                int(max(ys)),
            )
            regions.append(
                OCRRegion(
                    bbox=bbox,
                    text=text,
                    confidence=float(confidence),
                    rect=rect,
                )
            )

        return sorted(regions, key=lambda region: (region.rect[1], region.rect[0]))

    def choose_mode(self, image: Image.Image, regions: list[OCRRegion]) -> str:
        if not regions:
            return "easyocr_paragraph"

        mean_confidence = sum(region.confidence for region in regions) / len(regions)
        wide_regions = [
            region
            for region in regions
            if region.confidence >= 0.5 and region.width >= image.width * 0.45 and region.height >= 14
        ]
        if len(regions) >= 20 and mean_confidence < 0.55 and len(wide_regions) <= 2:
            return "trocr_regions"

        if len(wide_regions) >= 6:
            return "easyocr_paragraph"

        if len(regions) >= 14:
            return "easyocr_paragraph"

        return "trocr_regions"

    def merge_regions_into_lines(self, regions: list[OCRRegion]) -> list[OCRLine]:
        filtered = [
            region
            for region in regions
            if region.confidence >= 0.25 and region.width >= 25 and region.height >= 12
        ]
        if not filtered:
            return []

        lines: list[OCRLine] = []
        current_x1, current_y1, current_x2, current_y2 = filtered[0].rect
        current_count = 1
        current_center_sum = (current_y1 + current_y2) / 2
        current_height_sum = current_y2 - current_y1

        for region in filtered[1:]:
            x1, y1, x2, y2 = region.rect
            current_center_y = current_center_sum / current_count
            region_center_y = (y1 + y2) / 2
            average_height = current_height_sum / current_count
            same_line = abs(region_center_y - current_center_y) <= max(average_height, region.height) * 0.65

            if same_line:
                current_x1 = min(current_x1, x1)
                current_y1 = min(current_y1, y1)
                current_x2 = max(current_x2, x2)
                current_y2 = max(current_y2, y2)
                current_count += 1
                current_center_sum += region_center_y
                current_height_sum += region.height
            else:
                lines.append(OCRLine(rect=(current_x1, current_y1, current_x2, current_y2), region_count=current_count))
                current_x1, current_y1, current_x2, current_y2 = region.rect
                current_count = 1
                current_center_sum = region_center_y
                current_height_sum = region.height

        lines.append(OCRLine(rect=(current_x1, current_y1, current_x2, current_y2), region_count=current_count))
        return lines

    def load_trocr(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        model_source = self.trocr_model_name
        if self.custom_model_path and self.custom_model_path.exists():
            model_source = str(self.custom_model_path)

        self.processor = TrOCRProcessor.from_pretrained(model_source)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_source).to(self.device)
        self.model.eval()
        self.model_source = model_source

    def recognize_with_easyocr_paragraph(self, image: Image.Image) -> str:
        rgb = np.array(image.convert("RGB"))
        paragraph_results = self.reader.readtext(rgb, detail=1, paragraph=True)
        chunks: list[str] = []

        for entry in paragraph_results:
            if len(entry) >= 2:
                chunks.append(entry[1])

        return cleanup_text(" ".join(chunks))

    def recognize_with_trocr(self, image: Image.Image, regions: list[OCRRegion]) -> str:
        self.load_trocr()
        assert self.processor is not None
        assert self.model is not None

        predictions: list[str] = []
        line_regions = self.merge_regions_into_lines(regions)
        candidates = line_regions or [OCRLine(rect=region.rect, region_count=1) for region in regions]
        if not candidates:
            candidates = [OCRLine(rect=(0, 0, image.width, image.height), region_count=1)]

        for line in candidates:
            x1, y1, x2, y2 = line.rect
            width = x2 - x1
            height = y2 - y1
            if width < 80 or height < 20:
                continue

            crop = image.crop(
                (
                    max(0, x1 - 12),
                    max(0, y1 - 10),
                    min(image.width, x2 + 12),
                    min(image.height, y2 + 10),
                )
            )

            if crop.height < 96:
                scale = max(1, int(np.ceil(96 / max(crop.height, 1))))
                crop = crop.resize((crop.width * scale, crop.height * scale))

            pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                token_ids = self.model.generate(pixel_values, generation_config=self.generation_config)

            text = self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
            if text:
                predictions.append(text)

        return cleanup_text(" ".join(predictions))

    def draw_boxes(self, image: Image.Image, regions: list[OCRRegion]) -> np.ndarray:
        canvas = np.array(image.convert("RGB")).copy()
        for region in regions:
            x1, y1, x2, y2 = region.rect
            color = (0, 180, 0) if region.confidence >= 0.6 else (255, 140, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        return canvas

    def run_image(self, image: Image.Image, force_mode: str = "auto") -> dict[str, object]:
        image = image.convert("RGB")
        regions = self.detect_regions(image)
        mode = self.choose_mode(image, regions) if force_mode == "auto" else force_mode
        average_confidence = (
            sum(region.confidence for region in regions) / len(regions) if regions else 0.0
        )

        if mode == "easyocr_paragraph":
            text = self.recognize_with_easyocr_paragraph(image)
        elif mode == "trocr_regions":
            text = self.recognize_with_trocr(image, regions)
            fallback_text = self.recognize_with_easyocr_paragraph(image)
            if prefers_easyocr_fallback(text, fallback_text):
                text = fallback_text
                mode = "easyocr_paragraph_fallback"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        annotated = self.draw_boxes(image, regions)
        return {
            "mode": mode,
            "text": text,
            "regions": regions,
            "average_confidence": average_confidence,
            "annotated_image": annotated,
        }

    def run(self, image_path: str, force_mode: str = "auto") -> dict[str, object]:
        image = Image.open(image_path).convert("RGB")
        return self.run_image(image, force_mode=force_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive OCR pipeline for printed or line-based images.")
    parser.add_argument("image_path", help="Path to the image to process.")
    parser.add_argument(
        "--mode",
        choices=["auto", "easyocr_paragraph", "trocr_regions"],
        default="auto",
        help="Force a specific OCR mode or let the script decide automatically.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-large-handwritten",
        help="Open-source Hugging Face TrOCR model to use when the TrOCR branch is active.",
    )
    parser.add_argument(
        "--custom-model-path",
        default=None,
        help="Optional local path to a fine-tuned TrOCR checkpoint. This is only used when explicitly provided.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where OCR text and annotated image will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = AdaptiveOCRPipeline(
        trocr_model_name=args.model_name,
        custom_model_path=args.custom_model_path,
    )
    result = pipeline.run(args.image_path, force_mode=args.mode)

    image_stem = Path(args.image_path).stem
    text_path = output_dir / f"{image_stem}_ocr.txt"
    annotated_path = output_dir / f"{image_stem}_annotated.png"

    text_path.write_text(str(result["text"]), encoding="utf-8")
    Image.fromarray(result["annotated_image"]).save(annotated_path)

    print(f"Mode: {result['mode']}")
    print(f"Detections: {len(result['regions'])}")
    if pipeline.model_source:
        print(f"Model source: {pipeline.model_source}")
    print(f"Text file: {text_path}")
    print(f"Annotated image: {annotated_path}")
    print()
    print(result["text"])


if __name__ == "__main__":
    main()
