import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import io
import argparse


def enhance_pdf(input_pdf: str, output_pdf: str, zoom: float = 2.0, contrast: float = 1.2, sharpness: float = 1.1):
    """
    Render each page of a PDF to an image, lightly enhance it, and write a new PDF.

    zoom:      render scale (2.0 = 2x resolution)
    contrast:  1.0 = no change, 1.2 = +20% (subtle)
    sharpness: 1.0 = no change, 1.1 = +10% (subtle)
    """
    src = fitz.open(input_pdf)
    out = fitz.open()

    mat = fitz.Matrix(zoom, zoom)

    for i in range(src.page_count):
        page = src[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap -> PIL image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Enhance contrast
        img = ImageEnhance.Contrast(img).enhance(contrast)

        # Enhance sharpness
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

        # PIL image -> PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Create a new page in the output PDF with the same size as the original page
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(page.rect, stream=img_bytes)

    out.save(output_pdf)
    out.close()
    src.close()


def main():
    parser = argparse.ArgumentParser(description="Lightly enhance a PDF and export a new PDF.")
    parser.add_argument("input", help="Path to input PDF (e.g., input.pdf)")
    parser.add_argument("output", help="Path to output PDF (e.g., output.pdf)")
    parser.add_argument("--zoom", type=float, default=2.0, help="Render scale (default: 2.0)")
    parser.add_argument("--contrast", type=float, default=1.2, help="Contrast factor (default: 1.2)")
    parser.add_argument("--sharpness", type=float, default=1.1, help="Sharpness factor (default: 1.1)")
    args = parser.parse_args()

    enhance_pdf(
        input_pdf=args.input,
        output_pdf=args.output,
        zoom=args.zoom,
        contrast=args.contrast,
        sharpness=args.sharpness,
    )
    print(f"Saved enhanced PDF to: {args.output}")


if __name__ == "__main__":
    main()
