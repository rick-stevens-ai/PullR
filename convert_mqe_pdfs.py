#!/usr/bin/env python3
"""
Batch convert PDFs in MQE directory to markdown using Nougat with GPU acceleration.
Run with: conda activate nougat && python convert_mqe_pdfs.py
"""

import os
import sys
from pathlib import Path
import subprocess
from datetime import datetime
import argparse

def find_pdfs(directory):
    """Find all PDF files in directory."""
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)

    pdfs = list(pdf_dir.glob("*.pdf"))
    return sorted(pdfs)

def get_output_path(pdf_path, output_dir):
    """Get the output markdown path for a PDF."""
    stem = pdf_path.stem
    return Path(output_dir) / f"{stem}.mmd"

def convert_pdf(pdf_path, output_dir, skip_existing=True):
    """Convert a single PDF using nougat."""
    output_path = get_output_path(pdf_path, output_dir)

    # Check if already converted
    if skip_existing and output_path.exists():
        return "skipped"

    try:
        # Run nougat on the PDF
        cmd = [
            "nougat",
            str(pdf_path),
            "-o", str(output_dir),
            "--no-skipping"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per PDF
        )

        if result.returncode == 0:
            # Nougat outputs .mmd files, rename to .md if desired
            return "success"
        else:
            return f"failed: {result.stderr[:100]}"

    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return f"error: {str(e)[:100]}"

def main():
    parser = argparse.ArgumentParser(description="Convert MQE PDFs to markdown using Nougat")
    parser.add_argument("--input-dir", default="MQE", help="Input directory with PDFs")
    parser.add_argument("--output-dir", default="MQE_markdown", help="Output directory for markdown files")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already converted files")
    parser.add_argument("--no-skip", action="store_false", dest="skip_existing", help="Reconvert all files")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to convert (for testing)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all PDFs
    pdfs = find_pdfs(args.input_dir)
    total = len(pdfs)

    if args.limit:
        pdfs = pdfs[:args.limit]
        print(f"Processing {len(pdfs)} PDFs (limited from {total} total)")
    else:
        print(f"Processing {total} PDFs")

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)

    # Process PDFs
    stats = {"success": 0, "skipped": 0, "failed": 0, "timeout": 0, "error": 0}
    failed_files = []

    start_time = datetime.now()

    for idx, pdf_path in enumerate(pdfs, 1):
        print(f"[{idx}/{len(pdfs)}] {pdf_path.name}...", end=" ", flush=True)

        result = convert_pdf(pdf_path, output_dir, args.skip_existing)

        if result == "success":
            stats["success"] += 1
            print("✓")
        elif result == "skipped":
            stats["skipped"] += 1
            print("⊘ (skipped)")
        elif result == "timeout":
            stats["timeout"] += 1
            failed_files.append((pdf_path.name, "timeout"))
            print("✗ (timeout)")
        elif result.startswith("failed"):
            stats["failed"] += 1
            failed_files.append((pdf_path.name, result))
            print(f"✗ {result}")
        else:
            stats["error"] += 1
            failed_files.append((pdf_path.name, result))
            print(f"✗ {result}")

    # Summary
    elapsed = datetime.now() - start_time
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {len(pdfs)}")
    print(f"  Success:       {stats['success']}")
    print(f"  Skipped:       {stats['skipped']}")
    print(f"  Failed:        {stats['failed']}")
    print(f"  Timeout:       {stats['timeout']}")
    print(f"  Error:         {stats['error']}")
    print(f"Time elapsed:    {elapsed}")

    if failed_files:
        print("\nFailed files:")
        for filename, reason in failed_files[:20]:  # Show first 20
            print(f"  - {filename}: {reason}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")

    print(f"\nOutput directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
