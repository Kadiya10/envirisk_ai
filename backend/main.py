import os

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from backend.llm.rag_llm import rag_llm
from backend.rag.embeddings import embedding_model
from backend.rag.vector_store import vector_store

from backend.database import (
    init_database,
    create_user,
    get_user_by_email
)

from backend.auth import (
    hash_password,
    verify_password
)


# ============================================================
# INITIALISATION DE L'APPLICATION
# ============================================================

app = FastAPI(
    title="EnviRisk AI",
    description=(
        "Système intelligent d'analyse environnementale "
        "et rédaction de rapports EIES/PGES"
    ),
)


# ============================================================
# INITIALISATION DE LA BASE DE DONNÉES
# ============================================================

init_database()


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
# MODÈLES DE DONNÉES
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "question"
    document_name: str | None = None


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


class ReportRequest(BaseModel):
    report_type: str
    question: str = ""


# ============================================================
# PAGES FRONTEND
# ============================================================

@app.get("/")
def read_root():
    return FileResponse(
        "frontend/index.html"
    )


@app.get("/app")
def application():
    return FileResponse(
        "frontend/app.html"
    )


@app.get("/login.html")
def login_page():
    return FileResponse(
        "frontend/login.html"
    )


@app.get("/register.html")
def register_page():
    return FileResponse(
        "frontend/register.html"
    )


# ============================================================
# AUTHENTIFICATION - INSCRIPTION
# ============================================================

@app.post("/register")
def register_user(
    request: RegisterRequest
):

    # Nettoyage des données
    name = request.name.strip()
    email = request.email.strip().lower()
    password = request.password

    # Vérification du nom
    if not name:
        raise HTTPException(
            status_code=400,
            detail="Le nom est obligatoire."
        )

    # Vérification de l'e-mail
    if not email:
        raise HTTPException(
            status_code=400,
            detail=(
                "L'adresse e-mail "
                "est obligatoire."
            )
        )

    # Vérification du mot de passe
    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail=(
                "Le mot de passe doit contenir "
                "au moins 8 caractères."
            )
        )

    # Vérifier si l'utilisateur existe déjà
    existing_user = get_user_by_email(
        email
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail=(
                "Un compte existe déjà "
                "avec cette adresse e-mail."
            )
        )

    # Hash du mot de passe
    password_hash = hash_password(
        password
    )

    # Création du compte
    user_id = create_user(
        name=name,
        email=email,
        password_hash=password_hash
    )

    return {
        "message": "Compte créé avec succès.",
        "user_id": user_id,
        "name": name,
        "email": email
    }


@app.post("/login")
def login_user(
    request: LoginRequest
):

    email = request.email.strip().lower()
    password = request.password

    if not email or not password:
        raise HTTPException(
            status_code=400,
            detail="L'e-mail et le mot de passe sont obligatoires."
        )

    user = get_user_by_email(email)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="E-mail ou mot de passe incorrect."
        )

    if not verify_password(
        password,
        user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="E-mail ou mot de passe incorrect."
        )

    return {
        "message": "Connexion réussie.",
        "user_id": user["id"],
        "name": user["name"],
        "email": user["email"]
    }


# ============================================================
# UTILITAIRE : CRÉATION DES CHUNKS
# ============================================================

def create_chunks(
    text: str,
    page: int,
    chunk_size: int = 500
):
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
                f"page_{page}_chunk_{start}"
            ),
            "text": " ".join(
                chunk_words
            ),
            "page": page,
        })

    return chunks


# ============================================================
# IMPORTATION DES DOCUMENTS PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    document_type: str = Form(...)
):

    document_type = (
        document_type
        .upper()
        .strip()
    )

    # Vérification du type de document
    if document_type not in [
        "EIES",
        "PGES",
        "AUTRE"
    ]:
        return {
            "message": (
                "Type de document invalide. "
                "Choisissez EIES, PGES ou AUTRE."
            )
        }

    # Vérification du fichier
    if not file.filename:
        return {
            "message": "Aucun fichier fourni."
        }

    # Vérification de l'extension
    if not file.filename.lower().endswith(
        ".pdf"
    ):
        return {
            "message": (
                "Veuillez importer uniquement "
                "un fichier PDF."
            )
        }

    # Création du dossier
    os.makedirs(
        "data/uploads",
        exist_ok=True
    )

    # Chemin du fichier
    file_path = os.path.join(
        "data/uploads",
        file.filename
    )

    # Lecture du fichier
    content = await file.read()

    # Enregistrement du PDF
    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            content
        )

    # Lecture du PDF
    reader = PdfReader(
        file_path
    )

    all_chunks = []

    # Extraction page par page
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

    # Aucun texte trouvé
    if not all_chunks:
        return {
            "message": (
                "Le PDF a été enregistré, "
                "mais aucun texte exploitable "
                "n'a été trouvé."
            ),
            "filename": file.filename,
            "document_type": document_type,
            "pages": len(reader.pages),
        }

    # Préparation des textes
    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    # Création des embeddings
    embeddings = embedding_model.encode(
        texts
    )

    # Enregistrement dans ChromaDB
    vector_store.add_chunks(
        chunks=all_chunks,
        embeddings=embeddings,
        document_name=file.filename,
        document_type=document_type,
    )

    return {
        "message": (
            "Document importé et indexé "
            "avec succès."
        ),
        "filename": file.filename,
        "document_type": document_type,
        "pages": len(reader.pages),
        "chunks": len(all_chunks),
        "total_chunks": vector_store.count(),
    }


# ============================================================
# QUESTIONS / ANALYSE RAG
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

    print(
        "DOCUMENT ACTIF :",
        request.document_name
    )

    result = rag_llm.answer(
        question=request.question,
        top_k=request.top_k,
        mode=request.mode,
        document_name=request.document_name
    )

    print(
        "RESULTAT RAG :",
        repr(result)
    )

    return result
