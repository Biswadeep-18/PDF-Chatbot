from PyPDF2 import PdfReader, PdfWriter
import io
from typing import List, Tuple

class PDFService:
    @staticmethod
    def extract_text(pdf_file: io.BytesIO, filename: str, max_pages: int = 10) -> Tuple[str, int]:
        """Extract text from a PDF file"""
        pdf_reader = PdfReader(pdf_file)
        text = ""
        num_pages = len(pdf_reader.pages)
        
        pages_to_process = min(num_pages, max_pages)
        
        for page_num in range(pages_to_process):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        if num_pages > max_pages:
            text += f"\n\n... [Note: Only first {max_pages} pages were processed to stay within AI limits] ..."
            
        return text, num_pages

    @staticmethod
    def merge_pdfs(pdf_contents: List[bytes]) -> bytes:
        """Merge multiple PDF byte contents into one"""
        pdf_writer = PdfWriter()
        
        for content in pdf_contents:
            pdf_reader = PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
        
        output = io.BytesIO()
        pdf_writer.write(output)
        output.seek(0)
        return output.getvalue()
