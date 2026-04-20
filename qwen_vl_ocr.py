from __future__ import annotations

import argparse
from pathlib import Path

import pdf2image
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OCR_PROMPT = (
    "Perform OCR on this image. Return only the extracted text in reading order. "
    "Preserve line breaks when they are visually meaningful. Do not add explanations."
)


def extract_pdf_images(pdf_path: Path, max_pages: int | None = None) -> list[Image.Image]:
    """Render PDF pages into RGB PIL images."""
    pdf_bytes = pdf_path.read_bytes()

    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(pdf_bytes)
        total_pages = len(document)
        page_limit = min(total_pages, max_pages) if max_pages else total_pages
        rendered_pages: list[Image.Image] = []

        for page_index in range(page_limit):
            page = document[page_index]
            bitmap = page.render(scale=150 / 72)
            rendered_pages.append(bitmap.to_pil().convert("RGB"))

        document.close()
        return rendered_pages
    except Exception as pdfium_error:
        try:
            rendered_pages = pdf2image.convert_from_bytes(pdf_bytes, dpi=150)
            rgb_pages = [page.convert("RGB") for page in rendered_pages]
            return rgb_pages[:max_pages] if max_pages else rgb_pages
        except Exception as pdf2image_error:
            raise RuntimeError(
                "Unable to render PDF pages. "
                f"pypdfium2 error: {pdfium_error}. "
                f"pdf2image error: {pdf2image_error}"
            ) from pdf2image_error


class QwenVLOCRPipeline:
    """OCR wrapper around Qwen2.5-VL using the official Transformers chat interface."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = self._select_torch_dtype()
        processor_kwargs: dict[str, int] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = self._load_model()

    def _select_torch_dtype(self) -> torch.dtype:
        if self.device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        model_kwargs: dict[str, object] = {"torch_dtype": self.torch_dtype}

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        if self.device == "cpu":
            model = model.to(self.device)
        model.eval()
        return model

    def recognize_image(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_OCR_PROMPT,
        max_new_tokens: int = 1024,
    ) -> str:
        rgb_image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[rgb_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""

    def run_image(
        self,
        image_path: str | Path,
        prompt: str = DEFAULT_OCR_PROMPT,
        max_new_tokens: int = 1024,
    ) -> dict[str, object]:
        image = Image.open(image_path).convert("RGB")
        text = self.recognize_image(image, prompt=prompt, max_new_tokens=max_new_tokens)
        return {
            "mode": "qwen2_5_vl",
            "text": text,
            "model_name": self.model_name,
            "device": self.device,
        }

    def run_pdf(
        self,
        pdf_path: str | Path,
        prompt: str = DEFAULT_OCR_PROMPT,
        max_new_tokens: int = 1024,
        max_pages: int | None = None,
    ) -> dict[str, object]:
        pages = extract_pdf_images(Path(pdf_path), max_pages=max_pages)
        page_texts: list[str] = []

        for index, page in enumerate(pages, start=1):
            page_text = self.recognize_image(page, prompt=prompt, max_new_tokens=max_new_tokens)
            page_texts.append(page_text)

        combined_text = "\n\n".join(
            f"page_{index:03d}\n{text}".rstrip() for index, text in enumerate(page_texts, start=1)
        ).strip()
        return {
            "mode": "qwen2_5_vl",
            "text": combined_text,
            "pages": page_texts,
            "model_name": self.model_name,
            "device": self.device,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR images or PDFs with Qwen2.5-VL.")
    parser.add_argument("input_path", help="Path to an image or PDF.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_QWEN_MODEL,
        help="Qwen2.5-VL model to load from Hugging Face.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_OCR_PROMPT,
        help="Instruction prompt sent to the model for OCR.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where OCR text files will be written.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional PDF page limit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per image or page.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional minimum pixel budget for the Qwen processor.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional maximum pixel budget for the Qwen processor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = QwenVLOCRPipeline(
        model_name=args.model_name,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    if input_path.suffix.lower() == ".pdf":
        result = pipeline.run_pdf(
            input_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_pages=args.max_pages,
        )
        combined_text_path = output_dir / f"{input_path.stem}_qwen_ocr.txt"
        combined_text_path.write_text(str(result["text"]), encoding="utf-8")

        for index, page_text in enumerate(result["pages"], start=1):
            page_text_path = output_dir / f"{input_path.stem}_page_{index:03d}_qwen_ocr.txt"
            page_text_path.write_text(page_text, encoding="utf-8")

        print(f"Mode: {result['mode']}")
        print(f"Model: {result['model_name']}")
        print(f"Device: {result['device']}")
        print(f"Pages: {len(result['pages'])}")
        print(f"Combined text file: {combined_text_path}")
        print()
        print(result["text"])
        return

    result = pipeline.run_image(
        input_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    text_path = output_dir / f"{input_path.stem}_qwen_ocr.txt"
    text_path.write_text(str(result["text"]), encoding="utf-8")

    print(f"Mode: {result['mode']}")
    print(f"Model: {result['model_name']}")
    print(f"Device: {result['device']}")
    print(f"Text file: {text_path}")
    print()
    print(result["text"])


if __name__ == "__main__":
    main()
