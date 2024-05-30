import streamlit as st
import rag
import os

TMP_DIR = "tmp/"
FILE_TYPES = ["pdf", "docx", "txt", "md"]
OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
OPENAI_BASE_URL = st.secrets["openai"]["OPENAI_BASE_URL"]

os.makedirs(TMP_DIR, exist_ok=True)

if "chroma_db" not in st.session_state:
    st.session_state["chroma_db"] = rag.connect_to_db()

st.header("RAG demo")

generator_tab, loader_tab = st.tabs(["Zadaj pytanie", "Dodaj treści"])

with loader_tab:
    with st.form("loader_form"):
        url = st.text_input("Wprowadź adres strony")

        uploaded_files = st.file_uploader("Załaduj plik", type=FILE_TYPES, accept_multiple_files=True)
        one, two = st.columns([5, 12])
        with one:
            if st.form_submit_button("Załaduj&nbsp;wskazane&nbsp;treści"):
                with st.spinner("Trwa ładowanie..."):
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            file_path = TMP_DIR + uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            rag.load_file(file_path)
                            os.remove(file_path)
                    if url:
                        rag.load_url(url)
        with two:
            if st.form_submit_button("Wyczyść kontekst"):
                with st.spinner("Trwa czyszczenie..."):
                    if st.session_state.docs_ids:
                        rag.clean_up(st.session_state.chroma_db, st.session_state.docs_ids)

    st.session_state["docs_ids"] = rag.get_stored_documents_ids(st.session_state.chroma_db)
    st.info("Liczba przechowywanych dokumentów: " + str(len(st.session_state["docs_ids"])))

with generator_tab:
    with st.form("generator_form"):
        query = (st.text_area("Wpisz pytanie"))

        if st.form_submit_button("Przetwarzaj"):
            if query == "":
                st.error("Nie podano pytania")
                valid = False
            else:
                with st.spinner("Trwa przetwarzanie..."):
                    answer = rag.retrieve(query, OPENAI_API_KEY, OPENAI_BASE_URL)
                    st.write(answer.content)
