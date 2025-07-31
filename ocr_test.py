#!/usr/bin/env python3
"""
Local OCR test script - Tests DocumentProcessor directly without API
"""

import sys
import os
import base64
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your document processor
try:
    from app.services.document_processor import DocumentProcessor, PDFType, OCR_AVAILABLE

    print(f"✅ DocumentProcessor imported successfully")
    print(f"   OCR Available: {OCR_AVAILABLE}")
except ImportError as e:
    print(f"❌ Failed to import DocumentProcessor: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_local_ocr(pdf_path: str = "ocr_test.pdf"):
    """Test OCR processing locally"""

    print("\n" + "=" * 60)
    print(f"LOCAL OCR TEST - {pdf_path}")
    print("=" * 60)

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    # Read the PDF
    print(f"\n1. Reading PDF file...")
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()

    print(f"   File size: {len(pdf_content):,} bytes")

    # Initialize DocumentProcessor
    print("\n2. Initializing DocumentProcessor...")
    doc_processor = DocumentProcessor(
        ocr_enabled=True,
        ocr_dpi=300,  # High quality
        min_text_threshold=100,
        parallel_ocr=True,
        max_workers=4
    )

    # Test PDF type detection
    print("\n3. Detecting PDF type...")
    pdf_type, initial_text, pages_needing_ocr = doc_processor.detect_pdf_type(pdf_content)

    print(f"   PDF Type: {pdf_type.value}")
    print(f"   Initial text extracted: {len(initial_text)} characters")
    print(f"   Pages needing OCR: {len(pages_needing_ocr)} pages")

    if pdf_type == PDFType.TEXT_BASED:
        print("   → This is a text-based PDF (no OCR needed)")
    elif pdf_type == PDFType.SCANNED:
        print("   → This is a scanned PDF (OCR required)")
    else:
        print(f"   → This is a mixed PDF (OCR needed for pages: {pages_needing_ocr})")

    # Process the PDF
    print("\n4. Processing PDF with smart extraction...")
    start_time = time.time()

    result = doc_processor.process_pdf_smart(pdf_content)

    print(f"\n   Extraction completed in {result.extraction_time:.2f} seconds")
    print(f"   Method used: {result.method}")
    print(f"   Pages processed: {result.pages_processed}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Text extracted: {len(result.text)} characters")

    if result.warnings:
        print(f"\n   ⚠️  Warnings:")
        for warning in result.warnings:
            print(f"      - {warning}")

    # Show sample of extracted text
    if result.text:
        print("\n5. Sample of extracted text:")
        print("-" * 40)
        # Show first 500 characters
        sample = result.text[:500].strip()
        if sample:
            print(sample)
            if len(result.text) > 500:
                print("\n[... truncated ...]")
        else:
            print("(No readable text found)")

    # Save full text for inspection
    output_file = pdf_path.replace('.pdf', '_extracted.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result.text)
    print(f"\n💾 Full extracted text saved to: {output_file}")

    # Test with base64 encoding (like the API would)
    print("\n6. Testing base64 processing (simulating API)...")
    base64_content = base64.b64encode(pdf_content).decode('utf-8')

    text, extraction_result = doc_processor.process_document(
        base64_content,
        "application/pdf"
    )

    print(f"   Base64 processing successful: {len(text)} characters extracted")

    return result


def test_ocr_on_text_image():
    """Create and test a simple image with text"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pytesseract

        print("\n" + "=" * 60)
        print("TESTING OCR ON GENERATED IMAGE")
        print("=" * 60)

        # Create a simple image with text
        print("\n1. Creating test image with text...")
        img = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(img)

        # Try to use a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            font = ImageFont.load_default()

        test_text = "MEDICARE APPEAL DECISION\nCase #12345\nStatus: APPROVED"
        draw.text((50, 50), test_text, fill='black', font=font)

        # Save image
        img.save('test_ocr_image.png')
        print("   Test image saved as: test_ocr_image.png")

        # Test OCR directly
        print("\n2. Testing OCR on image...")
        start_time = time.time()
        extracted_text = pytesseract.image_to_string(img)
        ocr_time = time.time() - start_time

        print(f"   OCR completed in {ocr_time:.2f} seconds")
        print(f"   Extracted text:")
        print("-" * 40)
        print(extracted_text)
        print("-" * 40)

    except ImportError as e:
        print(f"❌ Cannot run image test: {e}")


def check_ocr_dependencies():
    """Check all OCR dependencies"""
    print("\nOCR DEPENDENCY CHECK")
    print("-" * 40)

    # Python packages
    packages = {
        'pytesseract': 'OCR Python wrapper',
        'pdf2image': 'PDF to image conversion',
        'PIL': 'Image processing',
        'numpy': 'Array processing'
    }

    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {package:12} - {description}")
        except ImportError:
            print(f"❌ {package:12} - {description} (NOT INSTALLED)")

    # System commands
    print("\nSystem Dependencies:")
    import subprocess

    # Tesseract
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ Tesseract: {version}")

            # List available languages
            langs_result = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True)
            if langs_result.returncode == 0:
                langs = langs_result.stdout.strip().split('\n')[1:]  # Skip header
                print(f"   Languages: {', '.join(langs[:5])}{'...' if len(langs) > 5 else ''}")
    except FileNotFoundError:
        print("❌ Tesseract: NOT INSTALLED")

    # Poppler (pdfinfo)
    try:
        result = subprocess.run(['pdfinfo', '-v'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Poppler: Installed")
    except FileNotFoundError:
        print("❌ Poppler: NOT INSTALLED")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test OCR locally without API')
    parser.add_argument('pdf_file', nargs='?', default='ocr_test.pdf',
                        help='PDF file to test (default: ocr_test.pdf)')
    parser.add_argument('--check-deps', action='store_true',
                        help='Check OCR dependencies')
    parser.add_argument('--test-image', action='store_true',
                        help='Test OCR on a generated image')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for OCR (default: 300)')

    args = parser.parse_args()

    # Always check dependencies first
    check_ocr_dependencies()

    if not OCR_AVAILABLE:
        print("\n❌ OCR is not available. Install dependencies with:")
        print("   pip install pytesseract pdf2image Pillow numpy")
        print("   ./setup_ocr.sh  # For system dependencies")
        return

    # Run requested tests
    if args.test_image:
        test_ocr_on_text_image()

    # Test the PDF
    if os.path.exists(args.pdf_file):
        # Modify DPI if specified
        if args.dpi != 300:
            print(f"\nUsing custom DPI: {args.dpi}")

        result = test_local_ocr(args.pdf_file)

        # Summary
        if result:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            if result.method == 'ocr':
                print("✅ OCR was successfully used to extract text")
            elif result.method == 'mixed':
                print("✅ OCR was used for some pages")
            elif result.method == 'pypdf2':
                print("ℹ️  PDF was text-based, OCR not needed")
            else:
                print("❌ Extraction method unclear")
    else:
        print(f"\n❌ File not found: {args.pdf_file}")
        print("Please specify a valid PDF file to test")


if __name__ == "__main__":
    main()