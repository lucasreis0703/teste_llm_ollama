import streamlit as st
import fitz  # PyMuPDF
import base64
import os
import tempfile
import json
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# --- Configuração de UI ---
st.set_page_config(page_title="Data IA do Ndados", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #4B0082; color: white; }
    h1, h2, h3, p, div { color: white !important; }
    .stChatMessage { background-color: #5D3FD3; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Inicialização de Estado ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá. Sou a Data IA do Ndados. Tire dúvidas sobre ferramentas, documentações de projetos e propostas antigas."}
    ]
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# Configuração de Modelos Locais (Ollama)
LLM_TEXT = "llama3"
LLM_VISION = "llava"
EMBEDDING_MODEL = "nomic-embed-text"

def extract_pdf_content(uploaded_files):
    raw_text_data = []
    llava_llm = Ollama(model=LLM_VISION, temperature=0.1)
    
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        doc = fitz.open(tmp_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text_data.append(f"Doc: {file.name} | Pág {page_num}:\n{page.get_text('text')}")
            
            # Gargalo de Hardware mantido para extração de fluxogramas
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    vision_description = llava_llm.invoke(f"Analise estritamente a arquitetura de dados ou fluxograma da imagem. Descreva tecnicamente. [Processamento Backend Ativo]") 
                    raw_text_data.append(f"Visão Computacional (Pág {page_num}): {vision_description}")
                except Exception:
                    pass

        os.remove(tmp_path)
    return raw_text_data

def build_vector_db(text_chunks):
    from langchain_core.documents import Document
    docs = [Document(page_content=t) for t in text_chunks]
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(docs, embeddings)

def get_rag_chain(vectorstore):
    llm = Ollama(model=LLM_TEXT, temperature=0.1)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), memory=st.session_state["memory"]
    )

def route_intent(query, llm):
    """Roteador Semântico"""
    prompt = f"""Classifique a intenção do usuário em ESTRITAMENTE uma das três palavras:
    CALCULO - Se o usuário quer extrair preços, calcular ITIP ou analisar propostas comerciais.
    RAG - Se o usuário pergunta sobre escopo, ferramentas ou equipe de um projeto em PDF.
    GERAL - Se o usuário pede dicas de dados, IA, compara cenários hipotéticos de forma ampla ou tira dúvidas conceituais.
    Usuário: {query}
    Classificação:"""
    res = llm.invoke(prompt).strip().upper()
    if "CALCULO" in res: return "CALCULO"
    elif "RAG" in res: return "RAG"
    else: return "GERAL"

# --- Interface ---
col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists("foto.png"):
        st.image("foto.png", width=80)
with col2:
    st.title("Data IA do Ndados")

st.sidebar.header("Painel de Controle")
uploaded_files = st.sidebar.file_uploader("Upload de PDFs (Propostas/Docs)", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Indexar Documentos"):
    if uploaded_files:
        with st.spinner("Extraindo texto, inferindo visão e vetorizando..."):
            text_data = extract_pdf_content(uploaded_files)
            vs = build_vector_db(text_data)
            st.session_state["vectorstore"] = vs
            st.session_state["chain"] = get_rag_chain(vs)
            st.success("Base operacional. Memória carregada.")
    else:
        st.sidebar.error("Arquivo ausente.")

# --- Execução do Chat ---
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Insira sua diretriz técnica, dúvida ou solicitação de cálculo...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.spinner("Analisando rota de execução..."):
        llm_router = Ollama(model=LLM_TEXT, temperature=0.0)
        
        intent = route_intent(user_query, llm_router) if st.session_state["vectorstore"] else "GERAL"
        answer = ""

        # ROTA 1: Matemática e Extração Comercial
        if intent == "CALCULO":
            prompt_extracao = f"""Analise os documentos. Extraia as opções comerciais. Retorne ESTRITAMENTE um ARRAY JSON:
            [{{ "nome": "Nome", "preco_original": <float>, "preco_desconto": <float/null>, "semanas": <int>, "consultores": <int> }}]
            Query: {user_query}"""
            
            resposta_bruta = st.session_state["chain"].invoke({"question": prompt_extracao})["answer"]
            try:
                json_str = re.search(r'\[.*\]', resposta_bruta, re.DOTALL).group()
                propostas = json.loads(json_str)
                answer = "**Relatório Comercial e ITIP:**\n\n"
                
                for prop in propostas:
                    nome = prop.get("nome", "Indefinido")
                    preco_final = prop.get("preco_desconto") or prop.get("preco_original", 0)
                    semanas = prop.get("semanas", 1)
                    consultores = prop.get("consultores", 1)
                    
                    itip = preco_final / (semanas * consultores) if semanas > 0 and consultores > 0 else 0
                        
                    answer += f"### {nome}\n- **Valor Base Execução:** R$ {preco_final:,.2f}\n- **Prazo:** {semanas} semanas | **Alocação:** {consultores} consultores\n- **ITIP Determinado:** R$ {itip:,.2f} / sem-consultor\n\n"
            except Exception:
                answer = "Falha estrutural. O modelo não conseguiu isolar o JSON das propostas devido à complexidade do layout e limitações de formatação do PDF original."

        # ROTA 2: Busca Documental (RAG)
        elif intent == "RAG":
            response = st.session_state["chain"].invoke({"question": user_query})
            answer = response["answer"]

        # ROTA 3: Consultoria Livre
        else:
            prompt_geral = f"""Atue como um arquiteto de dados e IA sênior. Responda de forma direta, técnica e analítica. 
            Pergunta: {user_query}"""
            answer = llm_router.invoke(prompt_geral)
        
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)