#!/usr/bin/env python3
"""
FastAPI wrapper for PDF OCR extraction.
Upload a PDF and get Markdown text for each page.
"""
import os
import tempfile
import re
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict
import io

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

app = FastAPI(title="PDF OCR API", version="2.0.0")


# ----------------------------
# Helpers
# ----------------------------
def image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def save_enhanced_page_image(
    page: fitz.Page,
    out_path: Path,
    zoom: float = 1.7,
    contrast: float = 1.2,
    sharpness: float = 1.1,
    jpeg_quality: int = 88,
    max_side: int = 2200,
):
    """Render a PDF page -> PIL -> enhance -> save as JPG."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)


def make_client():
    load_dotenv()

    base_url = os.getenv("BASE_URL", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL", "").strip()

    if not base_url or not api_key or not model:
        raise SystemExit("Missing BASE_URL / API_KEY / MODEL in environment (.env).")

    if not base_url.endswith("/openai/v1/"):
        base_url = base_url.rstrip("/") + "/openai/v1/"

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def detect_script(text_sample: str) -> str:
    """Detect if text is primarily Cyrillic or Latin."""
    cyrillic_count = len(re.findall(r'[А-Яа-яЁёӘәІіҮүҒғҚқҢңҺһ]', text_sample))
    latin_count = len(re.findall(r'[A-Za-zƏəİıÖöÜüĞğŞşÇç]', text_sample))
    
    if cyrillic_count > latin_count:
        return "cyrillic"
    else:
        return "latin"


LATIN_PROMPT = (
    "Extract ALL text from this page in Markdown format.\n"
    "Rules:\n"
    "- Use proper Markdown formatting (headers with #, lists, bold, italic, etc.)\n"
    "- For images, use: ![](brief description of the image)\n"
    "- Preserve tables using Markdown table syntax\n"
    "- Keep chemical formulas as close as possible\n"
    "- Preserve all formatting, line breaks, and structure\n"
    "- Return ONLY the Markdown text, no explanations\n"
)

CYRILLIC_PROMPT = (
    "Extract ALL Cyrillic text from this page in Markdown format.\n"
    "This is Azerbaijani written in Cyrillic script (some letters differ from Russian).\n"
    "Rules:\n"
    "- Extract exactly what you see in Cyrillic, including special letters like Ә\n"
    "- Use proper Markdown formatting (headers with #, lists, bold, italic, etc.)\n"
    "- For images, use: ![](brief description of the image)\n"
    "- Preserve tables using Markdown table syntax\n"
    "- Keep chemical formulas as close as possible\n"
    "- Preserve all formatting, line breaks, and structure\n"
    "- Return ONLY the Markdown text in Cyrillic, no explanations\n"
)


def is_internal_server_error(exc: Exception) -> bool:
    """Check if exception is a 500 internal server error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    resp = getattr(exc, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    
    msg = str(exc).lower()
    
    if status == 500:
        return True
    if "internal server error" in msg:
        return True
    if "server error" in msg and ("500" in msg or "status: 500" in msg):
        return True
    
    return False


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception),
)
def extract_text_from_image(
    client: OpenAI, 
    model: str, 
    img_path: Path, 
    prompt: str, 
    max_tokens: int
):
    """Extract text from image using LLM."""
    img_data_url = image_to_data_url(str(img_path))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_data_url}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        if is_internal_server_error(e):
            raise
        raise


# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "PDF OCR API",
        "version": "2.0.0",
        "endpoints": {
            "/ocr": "POST - Extract text from PDF as Markdown (auto-detects script)"
        }
    }


@app.post("/ocr")
async def ocr_pdf(
    file: UploadFile = File(...),
    zoom: float = Form(1.7),
    contrast: float = Form(1.2),
    sharpness: float = Form(1.1),
    jpeg_quality: int = Form(88),
    max_side: int = Form(2200),
    max_tokens: int = Form(1500),
):
    """
    Extract text from PDF and return as Markdown for each page.
    Automatically detects Cyrillic vs Latin script.
    
    Returns:
        List[Dict]: [{"page_number": 1, "MD_text": "..."}, ...]
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / file.filename
        img_dir = temp_path / "images"
        img_dir.mkdir()
        
        # Save uploaded file
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get client
        try:
            client, model = make_client()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize client: {str(e)}")
        
        # Process PDF
        try:
            doc = fitz.open(str(pdf_path))
            results = []
            
            # First, sample first page to detect script
            if doc.page_count > 0:
                first_page = doc[0]
                sample_text = first_page.get_text()
                script_type = detect_script(sample_text)
                prompt = CYRILLIC_PROMPT if script_type == "cyrillic" else LATIN_PROMPT
            else:
                prompt = LATIN_PROMPT
            
            # Process each page
            for i in range(doc.page_count):
                page = doc[i]
                img_path = img_dir / f"page_{i+1:04d}.jpg"
                
                # Save enhanced image
                save_enhanced_page_image(
                    page=page,
                    out_path=img_path,
                    zoom=zoom,
                    contrast=contrast,
                    sharpness=sharpness,
                    jpeg_quality=jpeg_quality,
                    max_side=max_side,
                )
                
                # Extract text
                try:
                    md_text = extract_text_from_image(
                        client=client,
                        model=model,
                        img_path=img_path,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    md_text = f"[ERROR extracting page {i+1}: {str(e)}]"
                
                results.append({
                    "page_number": i + 1,
                    "MD_text": md_text
                })
            
            doc.close()
            
            return JSONResponse(content=results)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)