#!/usr/bin/env python3
"""
Extract markdown from PDFs using Nougat OCR
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_markdown_from_pdfs(pdf_dir, output_dir, batch_size=4, skip_existing=True):
    """
    Extract markdown from all PDFs in a directory using Nougat

    Args:
        pdf_dir: Directory containing PDFs
        output_dir: Directory to save markdown files
        batch_size: Number of PDFs to process in parallel (default: 4)
        skip_existing: Skip PDFs that already have markdown files
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all PDFs
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    if not pdf_files:
        print("No PDF files found!")
        return

    # Filter out already processed files if skip_existing is True
    if skip_existing:
        to_process = []
        for pdf_file in pdf_files:
            md_file = output_dir / f"{pdf_file.stem}.mmd"
            if not md_file.exists():
                to_process.append(pdf_file)
        print(f"Skipping {len(pdf_files) - len(to_process)} already processed PDFs")
        pdf_files = to_process

    if not pdf_files:
        print("All PDFs already processed!")
        return

    print(f"\nProcessing {len(pdf_files)} PDFs with Nougat...")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)

    # Process in batches
    successful = 0
    failed = 0
    failed_files = []

    for i in tqdm(range(0, len(pdf_files), batch_size), desc="Processing batches"):
        batch = pdf_files[i:i+batch_size]

        for pdf_file in batch:
            try:
                # Run nougat on single PDF
                # Note: nougat outputs files with .mmd extension by default
                cmd = [
                    "nougat",
                    str(pdf_file),
                    "-o", str(output_dir),
                    "--no-skipping"  # Process all pages
                ]

                # Run nougat with timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per PDF
                )

                if result.returncode == 0:
                    successful += 1
                else:
                    failed += 1
                    failed_files.append((pdf_file.name, result.stderr[:200]))

            except subprocess.TimeoutExpired:
                failed += 1
                failed_files.append((pdf_file.name, "Timeout (>5 minutes)"))
            except Exception as e:
                failed += 1
                failed_files.append((pdf_file.name, str(e)[:200]))

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")

    if failed_files:
        print("\nFailed files:")
        for fname, error in failed_files[:10]:  # Show first 10
            print(f"  - {fname}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    # Count output files
    md_files = list(output_dir.glob("*.mmd"))
    print(f"\nMarkdown files created: {len(md_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract markdown from PDFs using Nougat")
    parser.add_argument("pdf_dir", help="Directory containing PDF files")
    parser.add_argument("--output-dir", "-o", help="Output directory for markdown files (default: pdf_dir/markdown)")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Number of PDFs to process in parallel (default: 4)")
    parser.add_argument("--no-skip", action="store_true", help="Reprocess all PDFs even if markdown exists")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.pdf_dir, "markdown")

    extract_markdown_from_pdfs(
        args.pdf_dir,
        output_dir,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip
    )
