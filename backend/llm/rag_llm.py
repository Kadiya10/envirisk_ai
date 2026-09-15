from backend.llm.ollama import ollama_llm
from backend.rag.retriever import retriever


class RAGLLM:

    def answer(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "question"
    ) -> dict:

        # ==========================================================
        # RECHERCHE DES PASSAGES PERTINENTS
        # ==========================================================

        results = retriever.search(
            query=question,
            top_k=top_k,
            min_similarity=0.0
        )

        if not results:
            return {
                "answer": (
                    "Aucun passage pertinent n'a été trouvé "
                    "dans les documents."
                ),
                "document_types": [],
                "sources": [],
                "passages": [],
            }

        # ==========================================================
        # TYPES DE DOCUMENTS
        # ==========================================================

        document_types = []

        for result in results:

            doc_type = result.get(
                "document_type",
                "AUTRE"
            )

            if doc_type not in document_types:
                document_types.append(doc_type)

        document_types_text = ", ".join(
            document_types
        )

        # ==========================================================
        # CONSTRUCTION DU CONTEXTE
        # ==========================================================

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"""
PASSAGE {i}

Document : {result['document']}
Type de document : {result.get('document_type', 'AUTRE')}
Page : {result['page']}

{result['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # ==========================================================
        # RÈGLES GÉNÉRALES DE PRÉSENTATION
        # ==========================================================

        format_rules = """
RÈGLES DE PRÉSENTATION OBLIGATOIRES :

- Réponds en français.
- Structure clairement la réponse.
- Utilise des titres avec le format :
  ### Titre
- Utilise des sous-titres lorsque nécessaire.
- Utilise des listes à puces pour présenter plusieurs éléments.
- Utilise une numérotation lorsque les éléments doivent être ordonnés.
- Mets les informations importantes en évidence avec :
  **texte important**
- Sépare les différentes parties avec des lignes vides.
- Évite les longs blocs de texte difficiles à lire.
- Ne répète pas inutilement les mêmes informations.
- Termine, lorsque cela est pertinent, par les références
  aux documents et aux pages.
- N'utilise pas de HTML.
- N'utilise pas de tableau Markdown sauf si cela améliore
  réellement la compréhension.
"""

        # ==========================================================
        # MODE QUESTION
        # ==========================================================

        if mode == "question":

            instruction = f"""
Tu es EnviRisk AI, un assistant spécialisé
dans l'analyse des documents environnementaux.

Le ou les documents utilisés sont de type :
**{document_types_text}**

Réponds à la question en utilisant UNIQUEMENT
les informations présentes dans les passages fournis.

{format_rules}

IMPORTANT :

- N'invente aucune information.
- Ne complète pas les informations manquantes avec
  tes connaissances générales.
- Si les documents ne permettent pas de répondre,
  indique clairement :

  "Les documents fournis ne permettent pas de répondre
  avec certitude."

- Lorsque tu utilises une information importante,
  indique si possible le document et la page.
"""

        # ==========================================================
        # MODE RISQUES
        # ==========================================================

        elif mode == "risks":

            instruction = f"""
Tu es un expert en évaluation environnementale des projets.

Le ou les documents utilisés sont de type :
**{document_types_text}**

À partir UNIQUEMENT des passages fournis,
identifie les risques environnementaux mentionnés
ou clairement déductibles du document.

Pour chaque risque identifié, présente :

### 🔎 [Nom du risque]

- **Cause ou activité :** ...
- **Milieu ou population affecté :** ...
- **Conséquence possible :** ...
- **Niveau d'information :** ...
- **Référence :** document, page ...

{format_rules}

IMPORTANT :

- N'invente aucun risque.
- Utilise uniquement les informations présentes
  dans les passages.
- Si une information est absente, écris :
  **Non précisé dans le document.**
- Ne transforme pas une hypothèse en fait.
- Cite le document et la page lorsque c'est possible.
"""

        # ==========================================================
        # MODE IMPACTS
        # ==========================================================

        elif mode == "impacts":

            instruction = f"""
Tu es un expert en évaluation environnementale
et sociale (EIES).

Le ou les documents utilisés sont de type :
**{document_types_text}**

À partir UNIQUEMENT des passages fournis,
identifie les impacts environnementaux et sociaux
du projet.

Organise la réponse de manière professionnelle.

Pour chaque impact identifié, utilise cette structure :

### 🌱 [Intitulé de l'impact]

- **Type :** Impact environnemental ou social
- **Activité ou source :** ...
- **Milieu affecté :** ...
- **Population ou récepteurs concernés :** ...
- **Description :** ...
- **Conséquences possibles :** ...
- **Niveau d'information :** ...
- **Référence :** document, page ...

{format_rules}

IMPORTANT :

- N'invente aucune information.
- Utilise uniquement les passages fournis.
- Si une information est absente, écris :
  **Non précisé dans le document.**
- Distingue les impacts environnementaux
  des impacts sociaux lorsque le document
  le permet.
- Cite le document et la page lorsque c'est possible.
"""

        # ==========================================================
        # MODE EVALUATION
        # ==========================================================

        elif mode == "evaluation":

            instruction = f"""
Tu es un expert en évaluation environnementale
et sociale (EIES).

Le ou les documents utilisés sont de type :
**{document_types_text}**

À partir UNIQUEMENT des passages fournis,
analyse l'importance des impacts environnementaux
et sociaux mentionnés dans les documents.

Pour chaque impact identifié, utilise cette structure :

### 📊 [Intitulé de l'impact]

- **Activité ou source :** ...
- **Milieu affecté :** ...
- **Intensité :** ...
- **Étendue :** ...
- **Durée :** ...
- **Importance :** ...
- **Réversibilité :** ...
- **Justification :** ...
- **Référence :** document, page ...

{format_rules}

IMPORTANT :

- Utilise uniquement les informations présentes
  dans les passages.
- Ne crée aucune valeur absente du document.
- Si un critère n'est pas précisé, écris :
  **Non précisé dans le document.**
- Ne déduis pas une importance faible, moyenne
  ou forte si elle n'est pas indiquée ou clairement
  justifiée.
- Lorsque le document présente déjà une évaluation,
  reprends-la fidèlement.
- Cite le document et la page lorsque c'est possible.
"""

        # ==========================================================
        # MODE INCONNU
        # ==========================================================

        else:

            instruction = f"""
Tu es un assistant spécialisé dans l'analyse
environnementale.

Le ou les documents utilisés sont de type :
**{document_types_text}**

Réponds à la question en utilisant uniquement
les passages provenant des documents fournis.

{format_rules}

IMPORTANT :

- N'invente aucune information.
- Si les passages ne permettent pas de répondre,
  indique clairement :

  "Les documents fournis ne permettent pas de répondre
  avec certitude."
"""

        # ==========================================================
        # PROMPT FINAL
        # ==========================================================

        prompt = f"""
{instruction}

==================================================
QUESTION
==================================================

{question}

==================================================
PASSAGES DES DOCUMENTS
==================================================

{context}

==================================================
RÉPONSE
==================================================
"""

        # ==========================================================
        # APPEL À QWEN3
        # ==========================================================

        response = ollama_llm.generate(
            prompt
        )

        # ==========================================================
        # SOURCES
        # ==========================================================

        sources = []
        passages = []

        for result in results:

            sources.append({
                "document": result["document"],
                "document_type": result.get(
                    "document_type",
                    "AUTRE"
                ),
                "page": result["page"],
                "similarity_percent":
                    result["similarity_percent"],
            })

            passages.append({
                "document": result["document"],
                "document_type": result.get(
                    "document_type",
                    "AUTRE"
                ),
                "page": result["page"],
                "text": result["text"],
                "similarity_percent":
                    result["similarity_percent"],
            })

        # ==========================================================
        # RESULTAT FINAL
        # ==========================================================

        return {
            "answer": response,
            "document_types": document_types,
            "sources": sources,
            "passages": passages,
        }


rag_llm = RAGLLM()