import io
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import glob


async def extract_text_from_pdfs(files, session_id=None):
    """
    First processes uploaded PDFs using PyMuPDF to generate text and image files in the output directory.
    Then processes all .txt and .png files in the output_dir.
    It reads text from .txt files and processes images from .png files (e.g., OCR).
    Returns all text concatenated, and a list of processed file names.
    """
    text = ""
    pdf_names = []
    output_dir = f"pdf_output/{session_id or 'default'}"
    os.makedirs(output_dir, exist_ok=True)
    ocr_text = ""

    # First, process uploaded PDF files to generate text and image files
    for file in files:
        pdf_names.append(file.filename)
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        
        for page_num, page in enumerate(doc.pages()):
            # --- Extract selectable text ---
            page_text = page.get_text()
            text_file_path = os.path.join(output_dir, f"{file.filename}_page{page_num+1}_text.txt")
            
            if page_text.strip():
                with open(text_file_path, "w", encoding="utf-8") as f:
                    f.write(page_text)
                # --- If no selectable text, do OCR on the page image ---


                
            # --- Extract images from the page ---
            img_list = page.get_images(full=True)
            for img_num, img in enumerate(img_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                img_path = os.path.join(output_dir, f"{file.filename}_page{page_num+1}_img{img_num+1}.{ext}")
                with open(img_path, "wb") as img_file:
                    img_file.write(img_bytes)
        
        doc.close()

    # Now process all .txt files in the output_dir
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            file_text = f.read()
            text += file_text + "\n"

    # Process all .png files in the output_dir (OCR)
    png_files = glob.glob(os.path.join(output_dir, "*.jpeg"))
    for png_file in png_files:

        img = Image.open(png_file)
        # Convert to grayscale if needed
        gray_img = img.convert('L')
        ocr_result = pytesseract.image_to_string(gray_img)
        text += ocr_result + "\n"

    return ocr_text, text, pdf_names 