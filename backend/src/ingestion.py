import os
import re
import json
from pathlib import Path
from pypdf import PdfReader

def clean_text(text: str) -> str:
    """Removes irregular whitespaces, trailing breaks, and page header artifacts."""
    if not text:
        return ""
    # Replace multiple spaces/newlines with a single space or newline
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def extract_pdf_chunks(
    pdf_path: str,
    doc_source: str,
    fiscal_year: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> list[dict]:
    """
    Extracts pages from a PDF and produces text chunks with rich metadata.
    
    Metadata attached to every chunk:
      - chunk_id: unique identifier
      - source: e.g. "Pakistan Economic Survey"
      - fiscal_year: e.g. "2023-24"
      - chapter: filename/subject
      - page_number: exact source page
    """
    reader = PdfReader(pdf_path)
    file_name = Path(pdf_path).stem
    chunks = []
    chunk_counter = 0

    print(f"📖 Processing: {file_name} ({len(reader.pages)} pages)")

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        if not cleaned:
            continue

        # Simple character-level sliding window chunking with overlap
        start = 0
        while start < len(cleaned):
            end = start + chunk_size
            chunk_content = cleaned[start:end].strip()

            if len(chunk_content) > 50:  # Skip tiny noise fragments
                chunk_counter += 1
                chunks.append({
                    "id": f"{file_name}_p{page_num}_c{chunk_counter}",
                    "content": chunk_content,
                    "metadata": {
                        "source": doc_source,
                        "fiscal_year": fiscal_year,
                        "file_name": file_name,
                        "page_number": page_num,
                        "char_count": len(chunk_content)
                    }
                })

            start += (chunk_size - chunk_overlap)

    return chunks

if __name__ == "__main__":
    raw_dir = Path("backend/data/raw")
    processed_dir = Path("backend/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    # Map files to their domain metadata
    documents = [
        {
            "path": raw_dir / "pes_2023_24_growth.pdf",
            "source": "Pakistan Economic Survey",
            "year": "2023-24"
        },
        {
            "path": raw_dir / "pes_2023_24_money_and_credit.pdf",
            "source": "Pakistan Economic Survey",
            "year": "2023-24"
        }
    ]

    for doc in documents:
        if doc["path"].exists():
            doc_chunks = extract_pdf_chunks(
                str(doc["path"]),
                doc_source=doc["source"],
                fiscal_year=doc["year"]
            )
            all_chunks.extend(doc_chunks)

    output_file = processed_dir / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Total extracted chunks: {len(all_chunks)}")
    print(f"💾 Saved to: {output_file}")

