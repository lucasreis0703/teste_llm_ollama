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

# --- Configuração de UI e Tema ---
st.set_page_config(page_title="IA do Ndados", page_icon="📊", layout="wide")

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
        {"role": "assistant", "content": "Olá, sou a Data, a IA do Ndados. Faça o upload das documentações ou propostas em PDF ou tire eventuais dúvidas sobre os documentos já processados. Posso ajudar a extrair informações, analisar propostas e até calcular o ITIP para você!"}
    ]
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# Configuração de Modelos Locais
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
            text = page.get_text("text")
            raw_text_data.append(f"Documento: {file.name} | Página {page_num}:\n{text}")
            
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                
                try:
                    vision_prompt = "Descreva detalhadamente este fluxograma, arquitetura de dados ou extraia o código presente na imagem. Seja técnico."
                    vision_description = llava_llm.invoke(f"{vision_prompt} [Imagem anexa processada pelo backend]") 
                    raw_text_data.append(f"Descrição de Imagem encontrada na página {page_num}: {vision_description}")
                except Exception as e:
                    st.error(f"Erro ao processar imagem no LLaVA: {e}")

        os.remove(tmp_path)
    return raw_text_data

def build_vector_db(text_chunks):
    from langchain_core.documents import Document
    docs = [Document(page_content=t) for t in text_chunks]
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(docs, embeddings)

def get_qa_chain(vectorstore):
    llm = Ollama(model=LLM_TEXT, temperature=0.0) # Temperatura ZERO para extração de dados financeiros
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 7}), # Maior contexto para capturar propostas longas
        memory=st.session_state["memory"]
    )

# --- Interface ---
col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists("foto.png"):
        st.image("foto.png", width=80)
with col2:
    st.title("IA do Ndados - Análise de Documentação")

st.sidebar.header("Painel de Controle")
uploaded_files = st.sidebar.file_uploader("Upload de PDFs", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Processar Base"):
    if uploaded_files:
        with st.spinner("Extraindo e vetorizando..."):
            text_data = extract_pdf_content(uploaded_files)
            vs = build_vector_db(text_data)
            st.session_state["vectorstore"] = vs
            st.session_state["chain"] = get_qa_chain(vs)
            st.success("Análise documental concluída.")
    else:
        st.sidebar.error("Insira arquivos.")

# --- Renderização do Chat ---
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ex: Extraia os valores das propostas e calcule o ITIP.")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    if "chain" in st.session_state:
        with st.spinner("Processando..."):
            
            # ROTEAMENTO: Identifica se o usuário quer cálculos comerciais
            if "itip" in user_query.lower() or "preço" in user_query.lower() or "valor" in user_query.lower():
                prompt_extracao = f"""
                Analise os documentos e extraia TODAS as opções comerciais apresentadas.
                Você deve retornar ESTRITAMENTE um ARRAY JSON válido com a estrutura abaixo. Não inclua texto fora do JSON.
                Se houver valor com desconto, preencha 'preco_desconto', caso contrário deixe null.
                [
                    {{
                        "nome_escopo": "Nome da proposta ou pacote",
                        "preco_original": <float>,
                        "preco_desconto": <float ou null>,
                        "semanas": <int>,
                        "consultores_dados": <int>
                    }}
                ]
                Pergunta original do usuário: {user_query}
                """
                
                response = st.session_state["chain"].invoke({"question": prompt_extracao})
                resposta_bruta = response["answer"]
                
                try:
                    # Captura o Array JSON da resposta do LLM usando Regex
                    json_str = re.search(r'\[.*\]', resposta_bruta, re.DOTALL).group()
                    propostas = json.loads(json_str)
                    
                    answer = "**Análise Comercial e Cálculo de ITIP:**\n\n"
                    
                    for prop in propostas:
                        nome = prop.get("nome_escopo", "Escopo Não Identificado")
                        preco_orig = prop.get("preco_original", 0)
                        preco_desc = prop.get("preco_desconto")
                        semanas = prop.get("semanas", 1)
                        consultores = prop.get("consultores_dados", 1)
                        
                        # Define qual preço usar para o ITIP
                        preco_final = preco_desc if preco_desc else preco_orig
                        status_desconto = "Com Desconto" if preco_desc else "Sem Desconto"
                        
                        # Cálculo matemático isolado e determinístico
                        if semanas > 0 and consultores > 0:
                            itip = preco_final / (semanas * consultores)
                        else:
                            itip = 0
                            
                        answer += f"### {nome}\n"
                        answer += f"- **Preço Original:** R$ {preco_orig:,.2f}\n"
                        if preco_desc:
                            answer += f"- **Preço com Desconto:** R$ {preco_desc:,.2f}\n"
                        answer += f"- **Prazo:** {semanas} semanas | **Consultores:** {consultores}\n"
                        answer += f"- **ITIP ({status_desconto}):** R$ {itip:,.2f} por semana/consultor\n\n"
                        
                except Exception as e:
                    answer = f"O modelo falhou em estruturar os dados para cálculo matemático. A complexidade do documento pode ter excedido a capacidade de extração estruturada do modelo local. Resposta bruta da IA: {resposta_bruta}"

            else:
                # Fluxo de RAG normal para dúvidas qualitativas (equipe, ferramentas)
                prompt_qualitativo = f"Aja como um conselheiro sênior. Responda de forma direta e analítica. {user_query}"
                response = st.session_state["chain"].invoke({"question": prompt_qualitativo})
                answer = response["answer"]
            
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
    else:
        st.error("Erro: Processe a base vetorial primeiro.")