# OCR Text Extractor - Streamlit Application

A production-ready web application built with Streamlit for extracting text from images and PDF files using advanced AI-powered OCR technology powered by TrOCR (Transformer-based Optical Character Recognition).

## Features

### 📄 File Support
- **Images**: JPG, JPEG, PNG, GIF, WebP
- **PDFs**: Multi-page PDF support with configurable page processing limits

### 🚀 Core Features
- Real-time text extraction using TrOCR model
- GPU/CPU automatic detection and optimization
- Image preview with metadata (size, format, filename)
- Text statistics (character count, word count, line count)
- Download extracted text as TXT files
- Configurable image scaling for improved accuracy
- Progress indicators and real-time status updates
- Error handling and validation

### 🎨 UI/UX
- Professional modern design with purple gradient theme
- Responsive layout for desktop and mobile devices
- Smooth animations and hover effects
- Clean card-based layouts
- Success/Error/Info message styling
- Tab-based results for multi-page PDFs

## Project Structure

```
Oce_Custom_Model/
├── app.py                  # Main Streamlit application
├── styles.css              # External CSS styling
├── best_ocr_model/         # Pre-trained OCR model
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── processor_config.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── ocr_dataset/            # Training datasets (optional)
│   ├── written_name_train_v2.csv
│   ├── written_name_validation_v2.csv
│   ├── written_name_test_v2.csv
│   ├── train_v2/
│   ├── validation_v2/
│   └── test_v2/
├── test.ipynb              # Development notebook
└── .venv/                  # Virtual environment
```

## Installation

### Prerequisites
- Python 3.14+
- Virtual environment (recommended)

### Setup

1. **Clone/Navigate to project directory**:
   ```bash
   cd Oce_Custom_Model
   ```

2. **Create and activate virtual environment** (if not already done):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit torch transformers pillow pdf2image PyPDF2
   ```

   Or install from requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the App

1. **Upload a File**:
   - Click the "Choose File" button in the upload section
   - Select an image (JPG, PNG, GIF, WebP) or PDF file

2. **Adjust Settings** (Optional):
   - **Image Scale Factor**: Increase for better accuracy on small text (default: 2)
   - **Max PDF Pages**: Set how many pages to extract from PDFs (default: 5)

3. **View Results**:
   - Image preview with metadata displayed
   - Extracted text shown in styled result box
   - Statistics: Character count, Word count, Line count
   - Copy text from the text area

4. **Download Results**:
   - Click "Download Text (TXT)" to save extracted text

## Configuration

### Image Scaling
- **Scale Factor 1**: Original image size
- **Scale Factor 2-5**: Upscaled for better accuracy on small/blurry text

### PDF Processing
- Adjust "Max PDF Pages" to control processing speed
- Each page is processed separately with individual results

## Performance

### Device Support
- **GPU (CUDA)**: Automatic detection for NVIDIA GPUs
- **CPU**: Fallback for systems without GPU support

### Processing Time
- Typical image: 1-3 seconds
- PDF (5 pages): 5-15 seconds
- Varies based on image size, complexity, and hardware

## Technical Details

### Dependencies
- **streamlit**: Web framework
- **torch**: PyTorch for model inference
- **transformers**: HuggingFace transformers for TrOCR
- **pillow**: Image processing
- **pdf2image**: PDF to image conversion
- **PyPDF2**: PDF handling

### Model
- **TrOCR**: Transformer-based Optical Character Recognition
- **Pre-trained**: best_ocr_model (custom trained)
- **Input**: Images (RGB color)
- **Output**: Lowercase text strings

## Styling

CSS styling is loaded from `styles.css` and includes:
- Modern gradient design
- Responsive breakpoints for mobile/tablet/desktop
- Animation effects
- Color scheme: Purple (#667eea) and Navy (#764ba2)
- Accessibility-friendly contrast ratios

## Error Handling

The application includes comprehensive error handling for:
- Missing or corrupted files
- Unsupported file formats
- Model loading failures
- OCR processing errors
- PDF extraction issues

All errors are displayed to users with clear, actionable messages.

## Troubleshooting

### "Model not found" error
- Ensure `best_ocr_model` folder exists in the project directory
- Verify model files are present (config.json, model.safetensors, etc.)

### "CSS file not found" warning
- Ensure `styles.css` exists in the same directory as `app.py`
- App will use default Streamlit styling if CSS is missing

### Slow processing
- Reduce image scale factor or PDF page count
- Use GPU if available (automatic detection)
- Process smaller images first

### Memory issues
- Reduce max PDF pages
- Close other applications
- Use CPU-only mode if GPU causes issues

## Development

### Testing
```bash
# Test imports
python -c "import streamlit, torch, transformers, pdf2image; print('OK')"

# Validate syntax
python -m py_compile app.py
```

### Customization
- Edit `styles.css` to change colors and styling
- Modify `app.py` to adjust functionality
- Update configuration in settings section

## Performance Tips

1. **For better accuracy**:
   - Increase image scale factor to 3-5
   - Use high-quality, clear images
   - Ensure good lighting in photos

2. **For faster processing**:
   - Decrease image scale factor to 1-2
   - Process fewer PDF pages at once
   - Use GPU if available

3. **For production deployment**:
   - Use `~streamlit run app.py~ --logger.level=error` to reduce logs
   - Consider using Streamlit Cloud for hosting
   - Implement caching for repeated uploads

## License

This project uses pre-trained OCR models and follows their respective licenses.

## Support

For issues or questions, refer to:
- Streamlit Documentation: https://docs.streamlit.io
- Transformers Documentation: https://huggingface.co/docs/transformers
- PyTorch Documentation: https://pytorch.org/docs

## Version

- **App Version**: 1.0
- **Release Date**: April 14, 2026
- **Python**: 3.14+
- **Streamlit**: 1.56.0+
