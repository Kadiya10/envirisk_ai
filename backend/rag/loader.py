from pathlib import Path
from typing import List, Dict

import pymupdf # PyMuPDF


def extract_pdf_text(file_path: str) -> List[Dict]:
    """
    Extrait le texte d'un fichier PDF page par page.

    Retourne une liste contenant :
    - le numéro de page
    - le texte extrait
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier PDF n'existe pas : {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Le fichier fourni doit être un PDF."
        )

    pages = []

    document = pymupdf.open(path)

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if text and text.strip():
                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )
    finally:
        document.close()

    return pages