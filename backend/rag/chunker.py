from typing import List, Dict


def create_chunks(
    pages: List[Dict],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Découpe le texte extrait des pages en petits morceaux
    adaptés à la recherche RAG.

    Chaque chunk conserve :
    - le numéro de page
    - le texte
    - un identifiant unique
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être supérieur à 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap ne peut pas être négatif.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap doit être inférieur à chunk_size."
        )

    chunks = []
    chunk_id = 0

    for page in pages:

        page_number = page["page"]
        text = page["text"].strip()

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(start + chunk_size, text_length)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "page": page_number,
                    "text": chunk_text,
                })

                chunk_id += 1

            if end >= text_length:
                break

            start = end - chunk_overlap

    return chunks