import streamlit as st
import gc
import torch
import requests
import asyncio

from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# --- KÖRNYEZETVÉDELEM STREAMLITHEZ ---
# Ha a Streamlit eldobja a hurkot, azonnal csinálunk egy újat a szálhoz
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- KONFIGURÁCIÓ ---
st.set_page_config(page_title="Tanár", page_icon="🎓", layout="wide")

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        requests.post("http://localhost:11434/api/generate", 
                      json={"model": "gemma3:12b", "keep_alive": 0})
    except:
        pass

with st.sidebar:
    st.header("⚙️ Rendszervezérlő")
    if st.button("🗑️ GPU Memória Ürítése"):
        flush_memory()
        st.success("VRAM felszabadítva!")
    st.markdown("---")
    st.markdown("**Modell:** Gemma 3 (12B)")
    st.markdown("**Embedding:** BGE-M3")

st.title("Ágens-Alapú Magántanár")

# =====================================================================
# 1. CSAK A NEHÉZ MUNKÁT CACHE
@st.cache_resource
def build_vector_index():
    # Itt egyszer beállítjuk az embeddinget az indexeléshez
    Settings.embed_model = OllamaEmbedding(model_name="bge-m3", request_timeout=360.0)
    try:
        documents = SimpleDirectoryReader("./MD").load_data()
        return VectorStoreIndex.from_documents(documents)
    except Exception as e:
        st.error(f"Fájl hiba: Nem találom az MD mappát, vagy üres! ({e})")
        return None

with st.spinner('Könyvtár indexelése... (Ez csak egyszer fut le!)'):
    index = build_vector_index()

if not index:
    st.stop()

# =====================================================================
# 2. AZ ÁGENST MINDEN FUTÁSNÁL FRISS
# Így biztosan nem hivatkoznak egy halott aszinkron hurokra.

Settings.llm = Ollama(model="gemma3:12b", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="bge-m3", request_timeout=360.0)

query_engine = index.as_query_engine(similarity_top_k=4)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="tankonyv_kereso",
            description="Használd ezt a szerszámot minden ténybeli, irodalmi vagy történelmi kérdésnél!"
        ),
    )
]

system_prompt = (
    "Te egy magyar bölcsészprofesszor és precíz magántanár vagy. "
    "A feladatod: a kapott tankönyvi részletekből készíts összefoglalót. "
    "Mielőtt válaszolnál, MINDIG kutass a tudásbázisodban a pontos adatokért a 'tankonyv_kereso' szerszámmal! "
    "SZIGORÚ SZABÁLYOK:\n"
    "1. KIZÁRÓLAG helyes, választékos magyar nyelven válaszolj!\n"
    "2. TILOS az angolos tükörfordítás.\n"
    "3. TILOS más nyelvek szavait használni.\n"
    "4. A válaszod legyen strukturált, használj alcímeket és pontokba szedett listákat.\n"
    "5. Úgy fogalmazz, mintha egy érettségi vázlatot készítenél.\n"
    "6. Ha nincs információ a könyvben, mondd meg őszintén, ne találj ki semmit!\n"
    "7. A végén MINDIG tegyél fel egy ellenőrző kérdést a diáknak!"
)

# A friss ágens
agent = ReActAgent(
    tools=tools,
    llm=Settings.llm,
    system_prompt=system_prompt,
    verbose=True
)

# =====================================================================
# 3. CHAT UI ÉS BIZTONSÁGOS FUTTATÁS
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# A memória kontextusa
if "agent_ctx" not in st.session_state:
    st.session_state.agent_ctx = Context(agent)

# Eddigi üzenetek kirajzolása
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ASZINKRON BURKOLÓ FÜGGVÉNY ---
async def get_agent_response(msg):
    return await agent.run(user_msg=msg, ctx=st.session_state.agent_ctx)

# Új kérdés kezelése
if prompt := st.chat_input("Tedd fel a kérdésed a tankönyvvel kapcsolatban..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("A Tanár a válaszon gondolkodik és kutat a könyvben... (GPU 🚀)"):
            try:
                # ITT TÖRTÉNIK AZ ÁTTÖRÉS: A biztonságos hurokban hívjuk meg az aszinkron ágenst
                response = loop.run_until_complete(get_agent_response(prompt))
                
                if hasattr(response, "response"):
                    answer_text = str(response.response)
                else:
                    answer_text = str(response)
                
                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
            except Exception as e:
                st.error(f"Hiba a feldolgozás során: {e}")