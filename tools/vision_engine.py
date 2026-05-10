import os
import io
from pathlib import Path
from typing import Optional
from PIL import Image

class VisionEngine:
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initializes the Vision Engine for OCR and Image Analysis.
        :param tesseract_path: Optional path to the Tesseract executable (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe)
        """
        self.tesseract_path = tesseract_path
        if self.tesseract_path:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            except ImportError:
                pass

    def _get_pytesseract(self):
        try:
            import pytesseract
            return pytesseract
        except ImportError:
            return None

    def ocr_image(self, image: Image.Image) -> str:
        """Performs OCR on a PIL Image object."""
        pytesseract = self._get_pytesseract()
        if not pytesseract:
            return "[Error: pytesseract not installed]"
            
        try:
            # Basic preprocessing: convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            return f"[OCR Error: {str(e)}]"

    def extract_and_ocr_pdf(self, pdf_path: Path) -> str:
        """Extracts images from a PDF and runs OCR on them."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            ocr_results = []
            
            for i, page in enumerate(reader.pages):
                for j, image_obj in enumerate(page.images):
                    try:
                        # Convert pypdf image to PIL
                        img = Image.open(io.BytesIO(image_obj.data))
                        text = self.ocr_image(img)
                        if text:
                            ocr_results.append(f"--- IMAGE OCR (Page {i+1}, Img {j+1}) ---\n{text}")
                    except Exception:
                        continue
            
            return "\n\n".join(ocr_results)
        except Exception as e:
            return f"[PDF Vision Error: {str(e)}]"

    def extract_and_ocr_docx(self, docx_path: Path) -> str:
        """Extracts images from a Word document and runs OCR on them."""
        try:
            import docx
            doc = docx.Document(docx_path)
            ocr_results = []
            img_count = 0
            
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img_count += 1
                    try:
                        img_data = rel.target_part.blob
                        img = Image.open(io.BytesIO(img_data))
                        text = self.ocr_image(img)
                        if text:
                            ocr_results.append(f"--- IMAGE OCR (Img {img_count}) ---\n{text}")
                    except Exception:
                        continue
            
            return "\n\n".join(ocr_results)
        except Exception as e:
            return f"[Word Vision Error: {str(e)}]"
