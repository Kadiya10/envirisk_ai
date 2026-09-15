from backend.report.report_generator import report_generator


context = """
Document : KIN-ELENDA
Page : 45

Le projet prévoit des travaux susceptibles de générer
des poussières et du bruit pendant la phase de construction.

Document : KIN-ELENDA
Page : 46

Les populations riveraines peuvent être exposées aux
nuisances liées aux activités du chantier.
"""


print("========================================")
print("TEST GENERATION RAPPORT EIES")
print("========================================")

rapport = report_generator.generate(
    report_type="EIES",
    context=context
)

print("\nRAPPORT GÉNÉRÉ :\n")
print(rapport)