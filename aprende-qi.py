# =====================================================================================
# HACK: Corrige o problema do SQLite no Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =====================================================================================

import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
# --- MUDANÇA NA IMPORTAÇÃO ---
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

# --- CONFIGURAÇÃO INICIAL ---

DB_DIR = "chroma_db_persistent"

@st.cache_resource
def load_embedding_model():
    """Carrega o modelo de embedding SentenceTransformer."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- MUDANÇA NA FUNÇÃO load_llm ---
@st.cache_resource
def load_llm():
    """Carrega o modelo de linguagem generativo usando HuggingFaceEndpoint."""
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"] # Lendo o token explicitamente
    )
    return llm

embedding_model = load_embedding_model()
llm = load_llm() # Carregamos o modelo uma vez aqui

# --- O RESTO DO CÓDIGO PERMANECE IGUAL ---

def parse_txt(file):
    text_bytes = file.getvalue()
    try:
        return text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        st.warning(f"O arquivo '{file.name}' não estava em UTF-8. Tentando ler com codificação alternativa (latin-1).")
        return text_bytes.decode("latin-1", errors='ignore')

def parse_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def parse_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = ' '.join(t.strip() for t in soup.stripped_strings)
        return text
    except requests.RequestException as e:
        st.warning(f"Não foi possível acessar la URL: {url}. Erro: {e}")
        return None

def process_and_store_documents(files, urls):
    if not files and not urls:
        st.warning("Por favor, adicione arquivos ou URLs para iniciar o aprendizado.")
        return

    with st.spinner("Processando fontes e atualizando a base de conhecimento..."):
        all_texts = []
        for file in files:
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext == ".txt": all_texts.append(parse_txt(file))
            elif file_ext == ".pdf": all_texts.append(parse_pdf(file))
            elif file_ext == ".docx": all_texts.append(parse_docx(file))
        for url in urls:
            if url:
                text = parse_url(url)
                if text: all_texts.append(text)
        
        if not all_texts:
            st.error("Nenhum texto pôde ser extraído das fontes fornecidas.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.create_documents(all_texts)
        
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
        vector_db.persist()

    st.success(f"{len(documents)} trechos de texto foram adicionados à base de conhecimento com sucesso!")

def answer_question(query):
    if not os.path.exists(DB_DIR):
        st.error("A base de conhecimento está vazia. Por favor, adicione arquivos ou URLs primeiro.")
        return None, None

    with st.spinner("Buscando a resposta na base de conhecimento..."):
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain.invoke(query)
        answer = result['result']
        context_docs = result['source_documents']
        context = "\n".join([doc.page_content for doc in context_docs])
        return answer, context

def main():
    st.set_page_config(page_title="QA com Aprendizado Contínuo", layout="wide")
    st.title("🤖 Programa de QA com Aprendizado Persistente e Incremental")
    st.markdown("""
        Esta aplicação permite que você construa uma base de conhecimento a partir de múltiplos arquivos e URLs. 
        Depois, você pode fazer perguntas em linguagem natural e o sistema responderá com base no que aprendeu.
    """)
    with st.sidebar:
        st.header("📚 Tela de Aprendizado")
        st.markdown("Adicione aqui novas fontes de informação para o sistema.")
        uploaded_files = st.file_uploader(
            "Selecione múltiplos arquivos",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx'],
            help="Você pode arrastar e soltar vários arquivos de uma vez."
        )
        urls_input = st.text_area(
            "Informe URLs (uma por linha)",
            placeholder="https://site1.com\nhttps://site2.com/artigo",
            height=150
        )
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        if st.button("🧠 Processar e Aprender"):
            process_and_store_documents(uploaded_files, urls)
    st.header("❓ Tela de Perguntas")
    st.markdown("Faça uma pergunta com base no conteúdo que o sistema aprendeu.")
    user_query = st.text_input("Digite sua pergunta aqui:", "")
    if st.button("🔍 Obter Resposta"):
        if user_query:
            answer, context = answer_question(user_query)
            if answer:
                st.success(f"**Resposta:** {answer}")
                with st.expander("Ver contexto utilizado para a resposta"):
                    st.write(context)
        else:
            st.warning("Por favor, digite uma pergunta.")

if __name__ == "__main__":
    main()