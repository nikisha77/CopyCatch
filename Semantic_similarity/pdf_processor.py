import os
import fitz  # PyMuPDF
import PyPDF2
import logging
import re

logger = logging.getLogger(__name__)


class PDFProcessor:
    @staticmethod
    def extract_text_pymupdf(pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = "".join([page.get_text() for page in doc])
            doc.close()
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return ""

    @staticmethod
    def extract_text_pypdf2(pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return ""

    @classmethod
    def extract_text(cls, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            # Log and raise for orchestrator to handle
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = cls.extract_text_pymupdf(pdf_path)
        if not text or len(text.strip()) < 100:  # Check for meaningful content
            logger.info(
                f"PyMuPDF extraction insufficient for {pdf_path}, trying PyPDF2."
            )
            text = cls.extract_text_pypdf2(pdf_path)

        if not text or len(text.strip()) < 100:
            logger.error(
                f"Failed to extract meaningful text from {pdf_path} using all methods."
            )
            raise ValueError(
                f"Failed to extract meaningful text from {pdf_path} using all methods."
            )

        # Basic cleaning common to this processor
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Minimal header/footer removal (can be enhanced)
        text = re.sub(
            r"(?i)(Page\s+\d+\s+of\s+\d+)|(^\s*\d+\s*$)", "", text, flags=re.MULTILINE
        )
        # Fix common OCR errors like ligatures (can be expanded)
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
        # Attempt to dehyphenate words split across lines
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        return text.strip()
