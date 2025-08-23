import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# A linha abaixo foi corrigida para usar a nova biblioteca e remover o aviso de desatualização.
from langchain_community.embeddings import SentenceTransformerEmbeddings # <-- ALTERADO
from transformers import pipeline

# --- CONFIGURAÇÃO INICIAL ---

# Define o diretório para persistência do banco de dados vetorial
DB_DIR = "chroma_db_persistent"

# Carrega o modelo de embedding uma única vez para otimização
# Este modelo converte texto em vetores numéricos
@st.cache_resource
def load_embedding_model():
    """Carrega o modelo de embedding SentenceTransformer."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Carrega o pipeline de perguntas e respostas uma única vez
@st.cache_resource
def load_qa_pipeline():
    """Carrega o pipeline de QA da Hugging Face."""
    return pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

embedding_model = load_embedding_model()
qa_pipeline = load_qa_pipeline()

# --- FUNÇÕES DE PROCESSAMENTO DE DADOS ---

def parse_txt(file):
    """Extrai texto de um arquivo .txt."""
    return file.getvalue().decode("utf-8")

def parse_pdf(file):
    """Extrai texto de um arquivo .pdf."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def parse_docx(file):
    """Extrai texto de um arquivo .docx."""
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_url(url):
    """Extrai texto de uma URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove tags de script e style para limpar o texto
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = ' '.join(t.strip() for t in soup.stripped_strings)
        return text
    except requests.RequestException as e:
        st.warning(f"Não foi possível acessar a URL: {url}. Erro: {e}")
        return None

def process_and_store_documents(files, urls):
    """
    Processa arquivos e URLs, extrai o texto, divide em chunks e armazena no ChromaDB.
    Esta função implementa o aprendizado incremental.
    """
    if not files and not urls:
        st.warning("Por favor, adicione arquivos ou URLs para iniciar o aprendizado.")
        return

    with st.spinner("Processando fontes e atualizando a base de conhecimento..."):
        all_texts = []
        
        # Processa arquivos
        for file in files:
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext == ".txt":
                all_texts.append(parse_txt(file))
            elif file_ext == ".pdf":
                all_texts.append(parse_pdf(file))
            elif file_ext == ".docx":
                all_texts.append(parse_docx(file))
        
        # Processa URLs
        for url in urls:
            if url:
                text = parse_url(url)
                if text:
                    all_texts.append(text)
        
        if not all_texts:
            st.error("Nenhum texto pôde ser extraído das fontes fornecidas.")
            return

        # Divide os textos em pedaços (chunks) menores para melhor processamento
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.create_documents(all_texts)
        
        # Cria ou carrega o banco de dados vetorial com persistência
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
        vector_db.persist()

    st.success(f"{len(documents)} trechos de texto foram adicionados à base de conhecimento com sucesso!")

# --- FUNÇÃO DE PERGUNTAS E RESPOSTAS ---

def answer_question(query):
    """
    Recebe uma pergunta, busca o contexto relevante no ChromaDB e retorna a resposta.
    """
    if not os.path.exists(DB_DIR):
        st.error("A base de conhecimento está vazia. Por favor, adicione arquivos ou URLs primeiro.")
        return None, None

    with st.spinner("Buscando a resposta na base de conhecimento..."):
        # Carrega o banco de dados persistente
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

        # Busca por documentos similares à pergunta (busca de contexto)
        # O 'k=5' busca os 5 chunks mais relevantes
        relevant_docs = vector_db.similarity_search(query, k=5)
        
        if not relevant_docs:
            return "Não encontrei informações relevantes sobre isso na minha base de conhecimento.", None

        # Concatena o conteúdo dos documentos relevantes para formar o contexto
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Usa o pipeline de QA para encontrar a resposta dentro do contexto
        result = qa_pipeline(question=query, context=context)
        return result['answer'], context

# --- INTERFACE GRÁFICA (STREAMLIT) ---

def main():
    st.set_page_config(page_title="QA com Aprendizado Contínuo", layout="wide")
    
    st.title("🤖 Programa de QA com Aprendizado Persistente e Incremental")
    st.markdown("""
        Esta aplicação permite que você construa uma base de conhecimento a partir de múltiplos arquivos e URLs. 
        Depois, você pode fazer perguntas em linguagem natural e o sistema responderá com base no que aprendeu.
    """)

    # --- TELA DE ENTRADA PARA APRENDIZADO (na barra lateral) ---
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

    # --- TELA DE ENTRADA PARA PERGUNTAS (área principal) ---
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