@app.post("/generate-report")
def generate_report(request: ReportRequest):

    print("========== GENERATION RAPPORT ==========")
    print("TYPE :", request.report_type)
    print("QUESTION :", request.question)

    if request.report_type not in ["EIES", "PGES"]:
        return {
            "error": "Le type de rapport doit être EIES ou PGES."
        }

    print("1 - Type valide")

    query = request.question

    if not query:
        query = (
            "impacts environnementaux sociaux projet "
            "mesures d'atténuation gestion environnementale "
            "surveillance suivi populations"
        )

    print("2 - Requête :", query)

    results = retriever.search(
        query=query,
        top_k=20,
        min_similarity=0.0
    )

    print("3 - Recherche terminée")
    print("Nombre de résultats :", len(results))

    if not results:
        return {
            "error": "Aucun contenu pertinent trouvé dans les documents."
        }

    print("4 - Résultats trouvés")

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
PASSAGE {i}

Document : {result['document']}
Page : {result['page']}

{result['text']}
"""
        )

    context = "\n\n".join(context_parts)

    print("5 - Contexte construit")
    print("Longueur contexte :", len(context))

    print("6 - Appel du générateur de rapport")

    rapport = report_generator.generate(
        report_type=request.report_type,
        context=context
    )

    print("7 - Rapport généré")
    print("Longueur rapport :", len(rapport))

    return {
        "report_type": request.report_type,
        "report": rapport,
        "sources": [
            {
                "document": result["document"],
                "page": result["page"],
                "similarity_percent": result["similarity_percent"]
            }
            for result in results
        ]
    }