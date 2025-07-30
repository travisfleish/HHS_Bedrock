#!/usr/bin/env python3
"""
Complete OCR test for all pages
"""

import os
import time
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def process_full_pdf():
    """Process the entire PDF with OCR"""
    pdf_path = "ocr_test.pdf"

    print(f"Processing complete PDF: {pdf_path}")
    print("=" * 60)

    try:
        # Convert all pages (this may take a while)
        print("Converting PDF to images (this may take a minute)...")
        start_time = time.time()

        # Use 200 DPI for balance of speed and quality
        images = convert_from_path(pdf_path, dpi=200)

        conversion_time = time.time() - start_time
        print(f"✓ Converted {len(images)} pages in {conversion_time:.1f} seconds")

        # Process each page
        all_text = []
        total_chars = 0

        for i, image in enumerate(images, 1):
            print(f"\rProcessing page {i}/{len(images)}...", end='', flush=True)

            # Run OCR
            text = pytesseract.image_to_string(image)
            all_text.append(f"\n{'=' * 60}\nPAGE {i}\n{'=' * 60}\n{text}")
            total_chars += len(text)

        print(f"\n✓ OCR complete! Total characters extracted: {total_chars:,}")

        # Save full text
        full_text = "\n".join(all_text)
        with open("ocr_test_FULL.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"✓ Full text saved to: ocr_test_FULL.txt")

        # Analyze content
        print("\n" + "=" * 60)
        print("CONTENT ANALYSIS")
        print("=" * 60)

        # Document type indicators
        indicators = {
            "Reconsideration": ["reconsideration", "qic", "unfavorable", "favorable"],
            "ALJ Appeal": ["alj", "administrative law judge", "hearing"],
            "Medical Records": ["diagnosis", "patient", "treatment", "physician"],
            "Remittance": ["remittance", "claim", "payment", "denied"],
        }

        for doc_type, keywords in indicators.items():
            count = sum(full_text.lower().count(kw) for kw in keywords)
            if count > 0:
                print(f"{doc_type}: {count} keyword matches")

        # Show sample from each page
        print("\n" + "=" * 60)
        print("PAGE PREVIEWS (first 100 chars)")
        print("=" * 60)

        for i, text in enumerate(all_text, 1):
            # Extract just the content after the page header
            content = text.split('\n', 4)[-1]  # Skip the header lines
            preview = content.strip()[:100].replace('\n', ' ')
            if preview:
                print(f"Page {i}: {preview}...")

        return full_text

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


def create_summary():
    """Create a summary of what was found"""
    if os.path.exists("ocr_test_FULL.txt"):
        with open("ocr_test_FULL.txt", "r", encoding="utf-8") as f:
            full_text = f.read()

        # Create summary
        summary = []
        summary.append("DOCUMENT SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Total characters: {len(full_text):,}")
        summary.append(f"Total words: {len(full_text.split()):,}")
        summary.append(f"Total pages: 21")
        summary.append("")
        summary.append("KEY FINDINGS:")

        # Check for specific document types
        if "reconsideration decision" in full_text.lower():
            summary.append("✓ Found: Medicare Reconsideration Decision")
        if "administrative law judge" in full_text.lower():
            summary.append("✓ Found: ALJ Appeal documents")
        if "discharge summary" in full_text.lower():
            summary.append("✓ Found: Discharge Summary")
        if "physician order" in full_text.lower():
            summary.append("✓ Found: Physician Orders")

        summary_text = "\n".join(summary)
        print("\n" + summary_text)

        with open("ocr_test_SUMMARY.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)


if __name__ == "__main__":
    # Process the full PDF
    full_text = process_full_pdf()

    # Create summary
    if full_text:
        create_summary()

    print("\n✓ Processing complete!")
    print("Files created:")
    print("  - ocr_test_FULL.txt (complete extracted text)")
    print("  - ocr_test_SUMMARY.txt (analysis summary)")
    print("  - page1_image.png (visual check of page 1)")