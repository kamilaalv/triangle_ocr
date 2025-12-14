#!/usr/bin/env python3
import os
import re
import base64
import mimetypes
import argparse
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance

from dotenv import load_dotenv
from openai import OpenAI

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ----------------------------
# Helpers
# ----------------------------
def safe_stem(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
    return s[:120] if len(s) > 120 else s


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
    """
    Render a PDF page -> PIL -> enhance -> save as JPG.
    max_side caps resolution to reduce upload size with minimal OCR loss.
    """
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # cap resolution (big speed win, very low OCR loss)
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


DEFAULT_PROMPT = (
    "Extract ALL Azerbaijani text from this page. "
    "It may be handwritten and similar to Turkish. "
    "Return ONLY the extracted text (no explanations). "
    "Preserve line breaks when possible.,"
    "If there are pictures write there is a picture."
    "Try to keep the formatting as close as possible. "
    "Try to keep the tables as close as possible.  "
    "Keep chemical formulas as close as possible."
)


# ----------------------------
# Retry only for Internal Server Error (500)
# ----------------------------
def is_internal_server_error(exc: Exception) -> bool:
    """
    Retry only when the server throws 500-level internal errors.
    Works across different OpenAI SDK exception shapes.
    """
    # Try common fields on SDK exceptions
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)

    # Some SDK errors store response with status_code
    resp = getattr(exc, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)

    # Some SDK errors store a body/message that includes "500"
    msg = str(exc).lower()

    if status == 500:
        return True

    # sometimes represented as 5xx internal / server error strings
    if "internal server error" in msg:
        return True

    # conservative: also retry on "server error" when it looks like 500
    if "server error" in msg and ("500" in msg or "status: 500" in msg):
        return True

    return False


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception),
)
def extract_text_from_image(client: OpenAI, model: str, img_path: Path, prompt: str, max_tokens: int):
    """
    Calls model; retries only if internal server error (500).
    """
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
        # only retry 500 internal server error; otherwise raise immediately
        if is_internal_server_error(e):
            raise
        raise  # no retry


# ----------------------------
# Pipeline
# ----------------------------
def process_pdf(
    pdf_path: Path,
    out_txt_dir: Path,
    out_img_root: Path,
    client: OpenAI,
    model: str,
    zoom: float,
    contrast: float,
    sharpness: float,
    jpeg_quality: int,
    max_side: int,
    max_tokens: int,
    keep_images: bool,
    prompt: str,
):
    pdf_stem = safe_stem(pdf_path.name)
    pdf_img_dir = out_img_root / pdf_stem
    pdf_img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    combined_parts = []

    total_pages = doc.page_count
    page_iter = tqdm(range(total_pages), desc=pdf_path.name, unit="page", leave=False)

    for i in page_iter:
        page = doc[i]
        img_path = pdf_img_dir / f"page_{i+1:04d}.jpg"

        # preprocess -> save JPG
        save_enhanced_page_image(
            page=page,
            out_path=img_path,
            zoom=zoom,
            contrast=contrast,
            sharpness=sharpness,
            jpeg_quality=jpeg_quality,
            max_side=max_side,
        )

        # extract text
        try:
            text = extract_text_from_image(
                client=client,
                model=model,
                img_path=img_path,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except Exception as e:
            text = f"[ERROR extracting page {i+1}: {e}]"
            print(f"\n[FAIL] {pdf_path.name}: page {i+1}/{total_pages}: {e}")

        combined_parts.append(f"===== {pdf_path.name} | Page {i+1}/{total_pages} =====\n{text}\n")
        page_iter.set_postfix_str(f"{i+1}/{total_pages}")

    doc.close()

    out_txt_dir.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_txt_dir / f"{pdf_stem}.txt"
    out_txt_path.write_text("\n".join(combined_parts).strip() + "\n", encoding="utf-8")
    print(f"[DONE] {pdf_path.name} -> {out_txt_path}")

    if not keep_images:
        for p in pdf_img_dir.glob("*.jpg"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            pdf_img_dir.rmdir()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="PDF folder -> preprocess pages -> send images -> one TXT per PDF")
    parser.add_argument("--pdf_dir", required=True, help="Folder containing PDFs")
    parser.add_argument("--out_txt_dir", default="out_txt", help="Output folder for .txt files")
    parser.add_argument("--out_img_dir", default="page_images", help="Folder for preprocessed page images")
    parser.add_argument("--keep_images", action="store_true", help="Keep generated page images")

    # Preprocess controls (defaults tuned for speed + minimal OCR loss)
    parser.add_argument("--zoom", type=float, default=1.7, help="Render zoom (default 1.7)")
    parser.add_argument("--contrast", type=float, default=1.2, help="Contrast factor (default 1.2)")
    parser.add_argument("--sharpness", type=float, default=1.1, help="Sharpness factor (default 1.1)")
    parser.add_argument("--jpeg_quality", type=int, default=88, help="JPEG quality (default 88)")
    parser.add_argument("--max_side", type=int, default=2200, help="Max image side px (default 2200)")

    # Model extraction
    parser.add_argument("--max_tokens", type=int, default=900, help="Max tokens per page")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise SystemExit(f"pdf_dir not found: {pdf_dir}")

    client, model = make_client()
    prompt = DEFAULT_PROMPT

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {pdf_dir}")

    for pdf_path in tqdm(pdfs, desc="PDFs", unit="pdf"):
        process_pdf(
            pdf_path=pdf_path,
            out_txt_dir=Path(args.out_txt_dir),
            out_img_root=Path(args.out_img_dir),
            client=client,
            model=model,
            zoom=args.zoom,
            contrast=args.contrast,
            sharpness=args.sharpness,
            jpeg_quality=args.jpeg_quality,
            max_side=args.max_side,
            max_tokens=args.max_tokens,
            keep_images=args.keep_images,
            prompt=prompt,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()