from datetime import datetime
from pathlib import Path
import warnings
import os

import streamlit as st
from PIL import Image

# Suppress pin_memory warning when no GPU is available (common in cloud environments)
warnings.filterwarnings("ignore", message=".*pin_memory.*accelerator.*")


# Cloud-safe path resolution
def get_app_dir() -> Path:
    """Get application directory, supporting both local and cloud environments."""
    try:
        return Path(__file__).resolve().parent
    except (NameError, TypeError):
        # In some cloud/notebook environments, __file__ may not be defined
        return Path.cwd()


APP_DIR = get_app_dir()
CSS_PATH = APP_DIR / "styles.css"
MODEL_DIR = APP_DIR / "best_ocr_model"
OUTPUTS_DIR = Path(os.getenv("OCR_OUTPUT_DIR", str(APP_DIR / "outputs")))
DEFAULT_TROCR_MODEL = "microsoft/trocr-large-handwritten"


st.set_page_config(
    page_title="OCR Text Extractor",
    page_icon=":page_facing_up:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Load the external stylesheet when available."""
    if CSS_PATH.exists():
        with CSS_PATH.open("r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("styles.css was not found. Using default Streamlit styling.")


@st.cache_resource
def load_pipeline(use_custom_model: bool):
    """Initialize the OCR pipeline once per Streamlit session."""
    from ocr_pipeline import AdaptiveOCRPipeline

    custom_model_path = str(MODEL_DIR) if use_custom_model and MODEL_DIR.exists() else None
    return AdaptiveOCRPipeline(
        trocr_model_name=DEFAULT_TROCR_MODEL,
        custom_model_path=custom_model_path,
    )


def preprocess_image(image: Image.Image, scale: int = 2, max_dimension: int | None = None) -> Image.Image:
    """Convert to RGB and optionally upscale for OCR."""
    rgb_image = image.convert("RGB")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    if scale > 1:
        rgb_image = rgb_image.resize(
            (rgb_image.width * scale, rgb_image.height * scale),
            resampling,
        )

    if max_dimension and max(rgb_image.size) > max_dimension:
        ratio = max_dimension / max(rgb_image.size)
        rgb_image = rgb_image.resize(
            (max(1, int(rgb_image.width * ratio)), max(1, int(rgb_image.height * ratio))),
            resampling,
        )

    return rgb_image


def extract_pdf_images(pdf_file, max_pages: int | None = None) -> list[Image.Image]:
    """Convert uploaded PDF bytes into PIL images using pypdfium2."""
    pdf_bytes = pdf_file.getvalue()

    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(pdf_bytes)
        total_pages = len(document)
        page_limit = min(total_pages, max_pages) if max_pages else total_pages
        pdf_images: list[Image.Image] = []

        for page_index in range(page_limit):
            page = document[page_index]
            bitmap = page.render(scale=150 / 72)
            pdf_images.append(bitmap.to_pil().convert("RGB"))

        document.close()
        return pdf_images
    except Exception as error:
        raise RuntimeError(
            f"Unable to render PDF pages with pypdfium2: {error}"
        ) from error


def build_output_dir(uploaded_name: str) -> Path:
    """Create a timestamped folder for OCR artifacts."""
    run_name = f"{Path(uploaded_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUTS_DIR / run_name
    for subdir in ("pages", "processed", "annotated", "text"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def save_result_artifacts(
    output_dir: Path,
    base_name: str,
    original_image: Image.Image,
    processed_image: Image.Image,
    result: dict[str, object],
) -> dict[str, Path]:
    """Persist OCR artifacts for later inspection."""
    original_path = output_dir / "pages" / f"{base_name}.png"
    processed_path = output_dir / "processed" / f"{base_name}_processed.png"
    annotated_path = output_dir / "annotated" / f"{base_name}_annotated.png"
    text_path = output_dir / "text" / f"{base_name}_ocr.txt"

    original_image.convert("RGB").save(original_path)
    processed_image.convert("RGB").save(processed_path)
    Image.fromarray(result["annotated_image"]).save(annotated_path)
    text_path.write_text(str(result["text"]), encoding="utf-8")

    return {
        "original_path": original_path,
        "processed_path": processed_path,
        "annotated_path": annotated_path,
        "text_path": text_path,
    }


def run_ocr_pipeline(
    pipeline,
    image: Image.Image,
    scale_factor: int,
    max_dimension: int | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    """Preprocess an image and run the shared OCR pipeline."""
    processed_image = preprocess_image(image, scale_factor, max_dimension=max_dimension)
    result = pipeline.run_image(processed_image, force_mode="auto")
    return processed_image, result


def render_metrics(result: dict[str, object]) -> None:
    """Display OCR summary metrics."""
    text = str(result["text"])
    line_count = len([line for line in text.splitlines() if line.strip()]) or (1 if text else 0)
    average_confidence = float(result.get("average_confidence", 0.0))

    metric_columns = st.columns(5)
    metric_columns[0].metric("Characters", len(text))
    metric_columns[1].metric("Words", len(text.split()))
    metric_columns[2].metric("Lines", line_count)
    metric_columns[3].metric("Detections", len(result["regions"]))
    metric_columns[4].metric("Avg confidence", f"{average_confidence:.2f}")

    st.caption(f"Pipeline mode: `{result['mode']}`")


def render_text_output(result: dict[str, object], download_name: str) -> None:
    """Render OCR text, metrics, and download controls."""
    text = str(result["text"])
    st.subheader("OCR Results")
    if text:
        st.text_area(
            "Extracted text",
            value=text,
            height=260,
            key=f"ocr_text_{download_name}",
        )
        with st.expander("Plain text preview", expanded=True):
            st.code(text, language="text")
    else:
        st.warning("OCR completed, but no text was returned for this file.")

    render_metrics(result)
    st.download_button(
        label="Download Text",
        data=text,
        file_name=download_name,
        mime="text/plain",
    )


load_css()

col_left, col_center, col_right = st.columns([0.2, 0.6, 0.2])
with col_center:
    st.markdown(
        """
        <div class="header-container">
            <h1>OCR Text Extractor</h1>
            <p>Extract text from images and PDFs using TrOCR with optional custom model switching.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

settings_col, info_col = st.columns([1, 1], gap="large")

with settings_col:
    st.markdown(
        """
        <div class="upload-section">
        <h2>Upload File</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=["jpg", "jpeg", "png", "gif", "webp", "pdf"],
        help="Supported formats: JPG, JPEG, PNG, GIF, WebP, PDF",
    )

with info_col:
    st.markdown(
        """
        <div class="sidebar-section">
        <h3>Settings</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scale_factor = st.slider(
        "Image Scale Factor",
        min_value=1,
        max_value=5,
        value=2,
        help="Upscale the image before OCR. Useful for small text.",
    )

    max_pdf_pages = st.number_input(
        "Max PDF Pages to Process",
        min_value=1,
        max_value=100,
        value=5,
        help="Limit how many PDF pages are processed in one run.",
    )

    use_custom_model = st.toggle(
        "Use custom TrOCR model",
        value=False,
        disabled=not MODEL_DIR.exists(),
        help="Switch from microsoft/trocr-large-handwritten to your local best_ocr_model checkpoint.",
    )

    st.markdown(
        """
        <div class="sidebar-section">
        <h3>Model Info</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not MODEL_DIR.exists():
        st.caption("Custom TrOCR checkpoint not found. Using the Hugging Face model.")

with st.spinner("Loading OCR pipeline..."):
    try:
        pipeline = load_pipeline(use_custom_model)
    except Exception as error:
        st.error(f"Failed to load OCR pipeline: {error}")
        st.info("If dependencies are missing, install them with `pip install -r requirements.txt`.")
        st.stop()

st.success("OCR pipeline loaded successfully.")
st.info(f"Processing device: {pipeline.device.upper()}")
active_model = MODEL_DIR.name if use_custom_model and MODEL_DIR.exists() else DEFAULT_TROCR_MODEL
st.caption(f"Active TrOCR model: `{active_model}`")


if uploaded_file is not None:
    st.markdown("<hr>", unsafe_allow_html=True)
    file_extension = Path(uploaded_file.name).suffix.lower()
    output_dir = build_output_dir(uploaded_file.name)
    st.caption(f"Artifacts will be saved in: `{output_dir}`")

    if file_extension == ".pdf":
        st.markdown(
            """
            <div class="results-section">
            <h2>Processing PDF</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            pdf_images = extract_pdf_images(uploaded_file, max_pages=max_pdf_pages)
            pdf_copy_path = output_dir / uploaded_file.name
            pdf_copy_path.write_bytes(uploaded_file.getvalue())
            page_results: list[tuple[str, dict[str, object], Image.Image, dict[str, Path]]] = []
            combined_text_parts: list[str] = []

            for index, pdf_image in enumerate(pdf_images, start=1):
                progress_bar.progress(index / len(pdf_images))
                status_text.text(f"Processing page {index} of {len(pdf_images)}...")

                processed_image, result = run_ocr_pipeline(
                    pipeline,
                    pdf_image,
                    scale_factor,
                    max_dimension=2400,
                )
                page_name = f"page_{index:03d}"
                saved_paths = save_result_artifacts(
                    output_dir,
                    page_name,
                    pdf_image,
                    processed_image,
                    result,
                )
                page_results.append((f"Page {index}", result, processed_image, saved_paths))
                page_text = str(result["text"]).strip()
                if page_text:
                    combined_text_parts.append(f"{page_name}\n{page_text}")

            progress_bar.empty()
            status_text.empty()

            combined_text = "\n\n".join(combined_text_parts)
            (output_dir / "text" / f"{Path(uploaded_file.name).stem}_combined.txt").write_text(
                combined_text,
                encoding="utf-8",
            )

            if page_results:
                st.markdown(
                    """
                    <div class="success-message">
                    PDF processing completed. Page images, annotations, and OCR text were saved to the output folder.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.subheader("Combined PDF OCR Text")
                if combined_text:
                    st.text_area(
                        "Combined extracted text",
                        value=combined_text,
                        height=280,
                        key=f"combined_pdf_text_{Path(uploaded_file.name).stem}",
                    )
                else:
                    st.warning("PDF pages were processed, but no combined text was produced.")

                tabs = st.tabs([page_name for page_name, _, _, _ in page_results])
                for tab, (page_name, result, processed_image, saved_paths) in zip(tabs, page_results):
                    with tab:
                        preview_col, result_col = st.columns([1, 1], gap="large")

                        with preview_col:
                            st.image(processed_image, caption=f"{page_name} input", width="stretch")
                            st.image(
                                result["annotated_image"],
                                caption=f"{page_name} detections",
                                width="stretch",
                            )
                            st.caption(f"Saved text: `{saved_paths['text_path'].name}`")

                        with result_col:
                            render_text_output(
                                result,
                                download_name=f"{Path(uploaded_file.name).stem}_{page_name.lower().replace(' ', '_')}_ocr.txt",
                            )
                if combined_text:
                    st.download_button(
                        label="Download Combined PDF Text",
                        data=combined_text,
                        file_name=f"{Path(uploaded_file.name).stem}_combined_ocr.txt",
                        mime="text/plain",
                    )
            else:
                st.error("No PDF pages could be processed.")

        except Exception as error:
            st.error(f"Error processing PDF: {error}")

    else:
        try:
            original_image = Image.open(uploaded_file)
            processed_image, result = run_ocr_pipeline(
                pipeline,
                original_image,
                scale_factor,
            )
            save_result_artifacts(
                output_dir,
                Path(uploaded_file.name).stem,
                original_image,
                processed_image,
                result,
            )

            left_col, right_col = st.columns([1, 1], gap="large")

            with left_col:
                st.markdown(
                    """
                    <div class="results-section">
                    <h2>Preview</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(original_image, caption="Uploaded image", width="stretch")
                st.markdown(
                    f"""
                    <div class="card">
                    <p><strong>Size:</strong> {original_image.size[0]} x {original_image.size[1]} px</p>
                    <p><strong>Format:</strong> {original_image.format or "Unknown"}</p>
                    <p><strong>File name:</strong> {uploaded_file.name}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right_col:
                st.markdown(
                    """
                    <div class="results-section">
                    <h2>Processing</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress_bar = st.progress(25)
                progress_bar.progress(100)

                st.image(
                    result["annotated_image"],
                    caption="Detected text regions",
                    width="stretch",
                )
                render_text_output(
                    result,
                    download_name=f"{Path(uploaded_file.name).stem}_ocr.txt",
                )
                progress_bar.empty()

        except Exception as error:
            st.error(f"Error processing image: {error}")


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);">
        <p style="color: #666; margin-bottom: 10px;"><strong>OCR Text Extractor</strong></p>
        <p style="color: #999; font-size: 0.9em;">Powered by Streamlit, EasyOCR, and TrOCR.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
