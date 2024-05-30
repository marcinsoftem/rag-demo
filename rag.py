import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

CONFIG_FILE = "config/config.ini"
PROFIL = ("oxygen")
CHAT_MODEL = "gpt-3.5-turbo"
# CHAT_MODEL = "gpt-4o"
CHROMA_PATH = "db/rag-db"
EMBEDING_MODEL = "all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

PROMPT_TEMPLATE_2 = """
DOCUMENT:
{context}
QUESTION:
{question}
INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn’t contain the facts to answer the QUESTION return {none}
"""

PROMPT_TEMPLATE_3 = """
DOKUMENT:
{context}
PYTANIE:
{question}
INSTRUKCJA:
Odpowiedz na PYTANIE korzystając z tekstu DOKUMENTU.
Opieraj swoją odpowiedź na faktach zawartych w DOKUMENCIE.
Jeżeli DOKUMENT nie zawiera faktów pozwalających odpowiedzieć na PYTANIE, zwróć {none}.
"""


def connect_to_db() -> None:
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDING_MODEL)
    db_chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db_chroma


def save_documents(pages: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDING_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)


def load_pdf(pdf_file_path: str) -> None:
    save_documents(PyPDFLoader(pdf_file_path).load())


def load_docx(file_path: str) -> None:
    save_documents(Docx2txtLoader(file_path).load())


def load_txt(file_path: str) -> None:
    save_documents(TextLoader(file_path).load())


def load_md(file_path: str) -> None:
    save_documents(UnstructuredMarkdownLoader(file_path).load())


def load_url(url: str) -> None:
    save_documents(SeleniumURLLoader([url]).load())


def load_file(file_path: str) -> None:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        load_pdf(file_path)
    elif ext == ".docx":
        load_docx(file_path)
    elif ext == ".txt":
        load_txt(file_path)
    elif ext == ".md":
        load_md(file_path)


def get_stored_documents_ids(chroma_db: Chroma) -> list[str]:
    return chroma_db.get().get("ids")


def clean_up(chroma_db: Chroma, ids: list[str]) -> None:
    chroma_db.delete(ids)


def retrieve(query: str, openai_api_key: str, base_url: str) -> str:
    db_chroma = connect_to_db()
    docs_chroma = db_chroma.similarity_search_with_score(query, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_3)
    prompt = prompt_template.format(context=context_text, question=query, none="Nie mam wiedzy na ten temat")
    print(prompt)
    model = ChatOpenAI(openai_api_key=openai_api_key, base_url=base_url, model=CHAT_MODEL)
    response_text = model.invoke(prompt)
    return response_text
