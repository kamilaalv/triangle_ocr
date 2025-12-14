
---

# FastAPI PDF OCR Extraction API

## Overview

This is a FastAPI wrapper for extracting text from PDF files using Optical Character Recognition (OCR). The API uploads a PDF and returns the text from each page in Markdown format. It automatically detects the script (Cyrillic or Latin) and applies the appropriate extraction model for each page.

## Features

* Upload a PDF file and get the text extracted as Markdown.
* Automatically detect whether the text is in Cyrillic or Latin script.
* Page images are processed and enhanced before text extraction.
* Supports PDF pages with text and images.
* Handles file uploads and enhances image quality for OCR.

## Requirements

1. Python 3.6+
2. Install required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```
3. A `.env` file with the following variables:

   * `BASE_URL`: The base URL for the OpenAI API.
   * `API_KEY`: Your OpenAI API key.
   * `MODEL`: The model you wish to use for text extraction.

## Endpoints

### 1. `GET /`

Returns a brief description of the API.

#### Response

```json
{
    "message": "PDF OCR API",
    "version": "2.0.0",
    "endpoints": {
        "/ocr": "POST - Extract text from PDF as Markdown (auto-detects script)"
    }
}
```

### 2. `POST /ocr`

Extracts text from the uploaded PDF and returns it in Markdown format for each page. The text will be extracted according to the detected script (Cyrillic or Latin).

#### Request Parameters

* **file**: The PDF file to be processed (must be `.pdf` format).
* **zoom**: The zoom factor for rendering images (default: 1.7).
* **contrast**: The contrast adjustment for the image (default: 1.2).
* **sharpness**: The sharpness adjustment for the image (default: 1.1).
* **jpeg_quality**: The JPEG quality for the page image (default: 88).
* **max_side**: Maximum side length for the page image (default: 2200).
* **max_tokens**: The maximum number of tokens for the text extraction (default: 1500).

#### Example Request:

```bash
curl -X 'POST' \
  'http://localhost:8000/ocr' \
  -F 'file=@yourfile.pdf' \
  -F 'zoom=1.7' \
  -F 'contrast=1.2' \
  -F 'sharpness=1.1' \
  -F 'jpeg_quality=88' \
  -F 'max_side=2200' \
  -F 'max_tokens=1500'
```

#### Response

Returns a list of dictionaries, where each dictionary contains the page number and the extracted Markdown text.

Example response:

```json
[
  {
    "page_number": 1,
    "MD_text": "# Header\nSome extracted content here..."
  },
  {
    "page_number": 2,
    "MD_text": "Some other page content"
  }
]
```

## OCR Extraction Logic

1. The uploaded PDF file is processed page by page.
2. Each page is converted to an enhanced image using a specified zoom, contrast, sharpness, and quality.
3. The image is sent to the OpenAI API (with the specified model) to extract the text in Markdown format.
4. The script type (Cyrillic or Latin) is automatically detected from the first page.
5. Each page's extracted content is returned in Markdown format, including images and tables.

## Error Handling

* If a non-PDF file is uploaded, the API returns an error with status code 400.
* If the extraction fails for a page, an error message for that page will be returned.

## Running the Server

To run the server, use the following command:

```bash
uvicorn app2:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

