from typing import Optional

from backend.llm.ollama import ollama_llm
from backend.rag.retriever import retriever


class RAGLLM:

    def answer(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "question",
        document_name: Optional[str] = None
    ) -> dict:

        results = retriever.search(
            query=question,
            top_k=top_k,
            min_similarity=0.0,
            document_name=document_name
        )

        if not results:
            return {
                "answer": (
                    "Aucun passage pertinent n'a été trouvé "
                    "dans le document sélectionné."
                ),
                "document_types": [],
                "sources": [],
                "passages": [],
            }

        context_parts = []

        for index, result in enumerate(results, start=1):

            context_parts.append(
                f"""
SOURCE {index}
Document : {result["document"]}
Type : {result["document_type"]}
Page : {result["page"]}
Pertinence : {result["similarity_percent"]} %

Passage :
{result["text"]}
"""
            )

        context = "\n".join(context_parts)

        # ====================================================
        # PROMPTS SELON LE MODE
        # ====================================================

        if mode == "risks":

            instruction = """
Identifie les principaux risques environnementaux
mentionnés ou déductibles du contexte fourni.

Pour chaque risque :
- donne son intitulé ;
- explique brièvement le risque ;
- indique la source et la page lorsque disponibles.

Ne crée pas d'informations qui ne sont pas appuyées
par le contexte.
"""

        elif mode == "impacts":

            instruction = """
Identifie les principaux impacts environnementaux
présents dans le contexte fourni.

Pour chaque impact :
- indique l'impact ;
- précise le milieu concerné ;
- explique brièvement l'effet ;
- indique la source et la page lorsque disponibles.

Ne crée pas d'informations qui ne sont pas appuyées
par le contexte.
"""

        elif mode == "evaluation":

            instruction = """
Évalue l'importance des impacts environnementaux
identifiés dans le contexte.

Pour chaque impact, présente :
- Impact ;
- Importance : faible, moyenne ou forte ;
- Justification ;
- Source et page lorsque disponibles.

L'évaluation doit rester cohérente avec les informations
présentes dans le contexte.
"""

        else:

            instruction = """
Réponds à la question de manière claire, précise
et professionnelle.

Utilise uniquement les informations présentes
dans le contexte fourni.

Lorsque c'est pertinent, cite le document et
le numéro de page.
"""

        # ====================================================
        # PROMPT FINAL
        # ====================================================

        prompt = f"""
Tu es EnviRisk AI, un assistant spécialisé
dans l'analyse environnementale des documents EIES
et PGES.

CONTEXTE DOCUMENTAIRE
=====================

{context}

INSTRUCTION
===========

{instruction}

QUESTION
========

{question}

Réponds en français.
"""

        # ====================================================
        # APPEL OLLAMA / QWEN
        # ====================================================

        response = ollama_llm.generate(
            prompt
        )

        # ====================================================
        # SOURCES
        # ====================================================

        sources = []

        for result in results:

            sources.append({
                "document": result["document"],
                "document_type": result["document_type"],
                "page": result["page"],
                "similarity_percent": result[
                    "similarity_percent"
                ],
            })

        # ====================================================
        # TYPES DE DOCUMENTS
        # ====================================================

        document_types = list(
            dict.fromkeys(
                result["document_type"]
                for result in results
            )
        )

        # ====================================================
        # PASSAGES
        # ====================================================

        passages = []

        for result in results:

            passages.append({
                "document": result["document"],
                "document_type": result["document_type"],
                "page": result["page"],
                "text": result["text"],
                "similarity_percent": result[
                    "similarity_percent"
                ],
            })

        # ====================================================
        # RESULTAT FINAL
        # ====================================================

        return {
            "answer": response,
            "document_types": document_types,
            "sources": sources,
            "passages": passages,
        }


# ============================================================
# INSTANCE RAG
# ============================================================

rag_llm = RAGLLM()