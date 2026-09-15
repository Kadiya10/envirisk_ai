from pathlib import Path
from typing import List, Dict

import pymupdf


def extract_pdf_text(file_path: str) -> List[Dict]:
    """
    Extrait le texte d'un fichier PDF page par page.

    Retourne une liste de dictionnaires contenant :
    - page : numéro de page
    - text : texte extrait
    """

    path = Path(file_path)

    # Vérifier que le fichier existe
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier PDF n'existe pas : {file_path}"
        )

    # Vérifier l'extension
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Le fichier fourni doit être un fichier PDF."
        )

    pages = []

    # Ouvrir le PDF
    document = pymupdf.open(path)

    try:
        # Lire chaque page
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            # Ignorer les pages sans texte
            if text and text.strip():

                pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

    finally:
        document.close()

    return pages