import os

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from backend.llm.rag_llm import rag_llm
from backend.rag.embeddings import embedding_model
from backend.rag.vector_store import vector_store


app = FastAPI(
    title="EnviRisk AI",
    description="Système intelligent d'analyse environnementale et rédaction de rapports EIES/PGES",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELE QUESTION
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "question"


# ============================================================
# MODELE RAPPORT
# ============================================================

class ReportRequest(BaseModel):
    report_type: str
    question: str = ""


# ============================================================
# ROUTE RACINE
# ============================================================

@app.get("/")
def read_root():

    return {
        "message": "EnviRisk AI est opérationnel"
    }


# ============================================================
# TEST API
# ============================================================

@app.get("/test")
def test_api():

    return {
        "message": "FastAPI fonctionne correctement"
    }


# ============================================================
# CREATION DES CHUNKS
# ============================================================

def create_chunks(
    text: str,
    page: int,
    chunk_size: int = 500
):
    """
    Découpe le texte d'une page en passages.
    """

    words = text.split()

    chunks = []

    for start in range(
        0,
        len(words),
        chunk_size
    ):

        chunk_words = words[
            start:start + chunk_size
        ]

        if not chunk_words:
            continue

        chunks.append({

            "id": (
                f"page_{page}_"
                f"chunk_{start}"
            ),

            "text": " ".join(
                chunk_words
            ),

            "page": page,

        })

    return chunks


# ============================================================
# IMPORT PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    document_type: str = Form(...)
):

    # ========================================================
    # NORMALISATION DU TYPE
    # ========================================================

    document_type = (
        document_type
        .upper()
        .strip()
    )


    # ========================================================
    # VALIDATION DU TYPE
    # ========================================================

    if document_type not in [
        "EIES",
        "PGES",
        "AUTRE"
    ]:

        return {

            "message":
                "Type de document invalide. "
                "Choisissez EIES, PGES ou AUTRE."

        }


    # ========================================================
    # VALIDATION PDF
    # ========================================================

    if not file.filename:

        return {

            "message":
                "Aucun fichier fourni."

        }


    if not file.filename.lower().endswith(".pdf"):

        return {

            "message":
                "Veuillez importer uniquement un fichier PDF."

        }


    # ========================================================
    # CREATION DOSSIER UPLOAD
    # ========================================================

    os.makedirs(
        "data/uploads",
        exist_ok=True
    )


    # ========================================================
    # CHEMIN FICHIER
    # ========================================================

    file_path = os.path.join(
        "data/uploads",
        file.filename
    )


    # ========================================================
    # LECTURE FICHIER
    # ========================================================

    content = await file.read()


    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            content
        )


    # ========================================================
    # LECTURE PDF
    # ========================================================

    reader = PdfReader(
        file_path
    )


    # ========================================================
    # EXTRACTION ET CHUNKS
    # ========================================================

    all_chunks = []


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()


        if not page_text:
            continue


        page_chunks = create_chunks(

            text=page_text,

            page=page_number

        )


        all_chunks.extend(
            page_chunks
        )


    # ========================================================
    # AUCUN TEXTE
    # ========================================================

    if not all_chunks:

        return {

            "message": (
                "Le PDF a été enregistré, "
                "mais aucun texte exploitable "
                "n'a été trouvé."
            ),

            "filename":
                file.filename,

            "document_type":
                document_type,

            "pages":
                len(reader.pages),

        }


    # ========================================================
    # PREPARATION TEXTES
    # ========================================================

    texts = [

        chunk["text"]

        for chunk in all_chunks

    ]


    # ========================================================
    # GENERATION EMBEDDINGS
    # ========================================================

    embeddings = embedding_model.encode(
        texts
    )


    # ========================================================
    # ENREGISTREMENT CHROMADB
    # ========================================================

    vector_store.add_chunks(

        chunks=all_chunks,

        embeddings=embeddings,

        document_name=file.filename,

        document_type=document_type,

    )


    # ========================================================
    # REPONSE
    # ========================================================

    return {

        "message":
            "Document importé et indexé avec succès.",

        "filename":
            file.filename,

        "document_type":
            document_type,

        "pages":
            len(reader.pages),

        "chunks":
            len(all_chunks),

        "total_chunks":
            vector_store.count(),

    }


# ============================================================
# QUESTION / RAG
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    print(
        "QUESTION RECUE :",
        request.question
    )

    print(
        "MODE RECUE :",
        request.mode
    )


    result = rag_llm.answer(

        question=request.question,

        top_k=request.top_k,

        mode=request.mode

    )


    print(
        "RESULTAT RAG :",
        repr(result)
    )


    return result