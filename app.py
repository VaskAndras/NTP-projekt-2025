import streamlit as st
import gc
import torch
import requests
import asyncio
import time
import json
import os
import threading
import importlib
try:
    psutil = importlib.import_module("psutil")
except ImportError:
    psutil = None
try:
    pynvml = importlib.import_module("pynvml")
except ImportError:
    pynvml = None
import csv
import pandas as pd
import plotly.graph_objects as go
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# =====================================================================
# 1. MEMÓRIA ÉS GPU KEZELÉS
# =====================================================================
def unload_model(model_name):
    try:
        requests.post("http://localhost:11434/api/generate", json={"model": model_name, "keep_alive": 0})
    except Exception:
        pass

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    unload_model("gemma3:12b")
    unload_model("racka-magantanar")

# =====================================================================
# 2. HÁTTÉRBEN FUTÓ HARDVER MONITOR

def hardware_monitor(stop_event, csv_filename="TDK_hardware_log.csv"):
    if psutil is None:
        print("CPU/RAM monitorozás nem indul: a psutil csomag nincs telepítve.")

    if pynvml is None:
        print("GPU monitorozás nem indul: a pynvml csomag nincs telepítve.")
        return

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"GPU monitorozás hiba (nem indul): {e}")
        return

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Idopont", "CPU_szazalek", "RAM_GB", "GPU_szazalek", "VRAM_GB", "GPU_Watt", "GPU_Celsius"])

        while not stop_event.is_set():
            try:
                now = datetime.now().strftime("%H:%M:%S")
                cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
                ram = psutil.virtual_memory().used / (1024**3) if psutil else 0.0
                
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram = info.used / (1024**3)
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_load = util.gpu
                
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                writer.writerow([now, f"{cpu:.1f}", f"{ram:.2f}", gpu_load, f"{vram:.2f}", f"{power:.1f}", temp])
                f.flush()
                
            except Exception as e:
                print(f"Monitorozási hiba futás közben: {e}")
            
            stop_event.wait(1.0)
            
    pynvml.nvmlShutdown()

# =====================================================================
# 3. ÉRTÉKELŐ RENDSZER

def evaluate_with_judge(user_prompt, retrieved_chunks, system_prompt, model_response, api_key, judge_model="openai/gpt-4o"):
    judge_prompt = f"""Te egy szigorú tudományos bíráló vagy egy oktatástechnológiai (EdTech/RAG) kutatásban.
Feladatod a generált tanári válasz objektív, szigorú kiértékelése három független dimenzióban (1-től 5-ig pontozva).

=== BEMENETEK ===
[FELHASZNÁLÓI KÉRDÉS]:{user_prompt}

[RAG KONTEXTUS (TANKÖNYVI FORRÁS)]:{retrieved_chunks}

[ELVÁRT SZEREP ÉS VISELKEDÉS (SYSTEM PROMPT)]:{system_prompt}

[MODELL ÁLTAL ADOTT VÁLASZ]:{model_response}

=== ÉRTÉKELÉSI METRIKÁK ÉS RUBRIKÁK ===

1. FAITHFULNESS (Tényhűség és Forrásfegyelem) [1-5]
- [5] Hibátlan: Kizárólag a forrásból dolgozik. Ha a forrásban nincs adat, vagy a kérdés anakronizmust tartalmaz (pl. vasút a Rákóczi-korban), helyesen elutasítja a válaszadást vagy leleplezi a hibát.
- [4] Megbízható: A kontextusra épül, minimális, nem zavaró háttértudással kiegészítve.
- [3] Vegyes: Valós tényeket ír, de a felét a saját belső súlyaiból vette, nem a kontextusból.
- [2] Hallucináció: Ténybeli tévedéseket állít, vagy készpénznek veszi az anakronisztikus csapdát.
- [1] Súlyos hiba: Teljes kitaláció vagy üres válasz.

2. ROLE_ADHERENCE (Perszóna és Szabálykövetés) [1-5]
SZIGORÚ NEGATÍV UTASÍTÁSOK VIZSGÁLATA:
- Ha a szerep Szókratészi volt, és a modell KÖZVETLENÜL MEGVÁLASZOLTA a kérdést ahelyett, hogy rávezetett volna, a pontszám MAX 2 lehet!
- Ha a szerep Tényszerű volt, de nincsenek forrásmegjelölések, a pontszám MAX 3 lehet!
- Ha a szerep Múzeumpedagógus volt, de nem hozott mai életből vett analógiát, a pontszám MAX 3 lehet!
- [5]: Maradéktalanul teljesíti a pedagógiai szerepet és a terjedelmi elvárásokat.
- [3]: Részben felveszi a szerepet, de megsért egy formai instrukciót.
- [1]: Teljesen figyelmen kívül hagyja a szerepet.

3. PEDAGOGICAL_UTILITY (Középiskolai Érthetőség és Struktúra) [1-5]
- [5] Kiváló tanári munka: Világos fogalmi struktúra, korosztályhoz illeszkedő nyelv, könnyen tanulható magyarázat.
- [3] Közepes: Érthető, de száraz, túl tömör vagy feleslegesen bonyolult körmondatokkal terhelt.
- [1] Használhatatlan: Zavaros gondolatmenet, nem segíti a megértést.

KIMENETI FORMÁTUM (KIZÁRÓLAG AZ ALÁBBI STRUKTÚRÁJÚ JSON):
{{
  "faithfulness_score": <int 1-5>,
  "faithfulness_reason": "<1-2 mondatos indoklás>",
  "role_adherence_score": <int 1-5>,
  "role_adherence_reason": "<1-2 mondatos indoklás a szabályok betartásáról>",
  "pedagogical_score": <int 1-5>,
  "pedagogical_reason": "<1-2 mondatos indoklás a tanári minőségről>",
  "composite_score": <float 1.0-5.0 átlag>
}}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "TDK-Benchmark"
    }
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        res_json = res.json()
        if "error" in res_json:
            return {"error": f"OpenRouter API hiba: {res_json['error']}"}
        content = res_json['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

# =====================================================================
# 4. ALAPBEÁLLÍTÁSOK ÉS INDEXELÉS

st.set_page_config(page_title="AI Benchmark és Magántanár chat", layout="wide")

@st.cache_resource
def get_reranker():
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=2,
        device="cpu"
    )

@st.cache_resource
def build_vector_index():
    Settings.llm = None 
    Settings.embed_model = OllamaEmbedding(model_name="bge-m3", request_timeout=360.0)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    
    def extract_metadata(file_path):
        fname = os.path.basename(file_path).lower()
        if "tori" in fname or "tortenelem" in fname or "tori" in file_path.lower():
            tantargy = "Történelem"
        elif "mir" in fname or "irodalom" in fname:
            tantargy = "Magyar Irodalom"
        elif "mny" in fname or "nyelvtan" in fname:
            tantargy = "Magyar Nyelvtan"
        else:
            tantargy = "Általános Tankönyv"
        return {"file_name": os.path.basename(file_path), "tantargy": tantargy}

    try:
        documents = SimpleDirectoryReader("./MD", file_metadata=extract_metadata).load_data()
        return VectorStoreIndex.from_documents(documents)
    except Exception as e:
        st.error(f"Hiba az indexelés során: {e}")
        return None

with st.spinner("Könyvtár indexelése optimalizált chunkolással..."):
    index = build_vector_index()
    reranker = get_reranker()

# =====================================================================
# 5. TESZTKÉRDÉSEK ÉS PERSZÓNÁK

PERSONAS = {
    "1. Tényszerű Érettségi Vizsgáztató (Szigorú RAG & Hivatkozás)":
        "Te egy szigorú érettségi vizsgáztató vagy. Kizárólag a megadott tankönyvi kontextusra támaszkodva válaszolj 6-8 mondatban. "
        "Minden lényegi állításod után zárójelben jelöld meg a forrást (pl. [Tankönyv]). "
        "TILOS külső tudásból kiegészíteni. Ha a megadott forrás nem tartalmaz elég adatot a válaszhoz vagy a kérdés anakronizmust tartalmaz, "
        "kizárólag ezt rögzítsd: 'A tankönyvi forrás alapján a kérdés nem válaszolható meg.'",

    "2. Gondolkodást segítő magántanár (Vezetett Rávezetés)":
        "Te egy támogató szókratészi mentor vagy. A célod, hogy a diák magától jöjjön rá az összefüggésekre. "
        "SOHA NE add meg a direkt választ a kérdésre! "
        "Írj egy 6-8 mondatos gondolatébresztő hátteret a kontextus alapján, világíts rá a kulcsfogalmakra, "
        "majd a szöveg legvégén tegyél fel egyetlen célzott, rávezető kérdést, ami gondolkodásra készteti a diákot. "
        "Ha a kérdéshez nincs adat a szövegben, kérdezz rá, hogy biztosan releváns forrást néz-e.",

    "3. Kritikai Elemző (Tévhit- és Anakronizmus-vadász)":
        "Te egy akadémiai forráskritikus vagy. A feladatod a kérdésben felvetett téma kritikai elemzése a források tükrében 6-8 mondatban. "
        "Különös figyelmet fordíts az időbeli tévedésekre (anakronizmusok), a forrás hiányosságaira és a történelmi/irodalmi tévhitekre. "
        "Minden állítást a szöveg pontos idézésével vagy parafrazeálásával támassz alá vagy cáfolj.",

    "4. Múzeumpedagógus (Modern Analógiák)":
        "Te egy modern múzeumpedagógus vagy. Magyarázd el a témát egy 15-16 éves középiskolásnak 6-8 mondatban. "
        "A kontextusban lévő tényeket kötelező 100%-os pontossággal megtartani, de a jelenséget kösd össze legalább két "
        "modernkori párhuzammal vagy analógiával (pl. közösségi média, modern hírközlés, mai társadalmi minták).",

    "5. Didaktikai Összefoglaló":
        "Te egy precíz egyetemi jegyzetíró vagy. Készíts egy tömör, didaktikus szintézist a felvetett problémáról. "
        "A válaszod tartalmazzon: 1. Egy 2-3 mondatos elméleti felvezetést, 2. Három pontba szedett ok-okozati összefüggést a forrásból, "
        "3. Egy 1 mondatos konklúziót. Ha nincs elég adat a forrásban, a hiányzó szempontokat tételesen sorold fel."
}

TEST_QUESTIONS = {
    "Töri [Ok-Okozat]: 1848 és a jobbágyfelszabadítás":
        "Milyen gazdasági és társadalmi kényszerek vezettek a jobbágyfelszabadítás követeléséhez az 1848-as forradalom előestéjén?",

    "Töri [Anakronizmus / Csapda]: Technológia a Rákóczi-szabadságharcban":
        "Hogyan befolyásolta a vasúthálózat és a távíró kiépültsége a Rákóczi-szabadságharc hadmozdulatait?",

    "Töri [Out-of-Domain / Elutasítás]: Churchill és Jalta":
        "Milyen kompromisszumos megállapodást kötött Churchill és Sztálin a jaltai konferencián a szivarimport vámjairól?",

    "Magyar [Irodalmi Elemzés]: Ady szimbolizmusa":
        "Milyen motívumokon keresztül ragadja meg a halál és a végzet témáját Ady Endre 'A fekete zongora' című versében?",

    "Magyar [Nyelvtan / Szövegtan]: Koherencia és anafora":
        "Hogyan biztosítják a névmások és a kötőszavak a szövegösszetartó erőt (koherenciát) egy érvelő szövegben?"
}

# =====================================================================
# 6. PDF GENERÁLÓ MOTOR
# =====================================================================
def generate_tdk_pdf_report(results_list, filename="TDK_Benchmark_Eredmenyek.pdf"):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Magyar ékezetes betűtípusok regisztrálása
    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    Story = []
    styles = getSampleStyleSheet()

    # Stílusok felülírása a magyar betűtípusra
    title_style = styles['Heading1']
    title_style.fontName = 'DejaVu-Bold'
    title_style.alignment = TA_LEFT
    title_style.spaceAfter = 20

    sub_title_style = styles['Heading2']
    sub_title_style.fontName = 'DejaVu-Bold'
    sub_title_style.spaceBefore = 15
    sub_title_style.spaceAfter = 5
    
    normal_style = styles['Normal']
    normal_style.fontName = 'DejaVu'
    normal_style.alignment = TA_JUSTIFY
    
    bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='DejaVu-Bold', spaceAfter=5)
    score_style = ParagraphStyle('ScoreStyle', parent=styles['Normal'], fontName='DejaVu-Bold', textColor='blue', spaceAfter=10)

    # Cím és Bevezető
    Story.append(Paragraph("TDK AI Benchmark Mérési Jegyzőkönyv", title_style))
    
    bevezeto = (
        "Ez a dokumentum az automatizált RAG benchmark kísérlet eredményeit tartalmazza a TDK dolgozathoz. "
        "A teszt során a modellek különböző tanári szerepekben (perszónákban) válaszoltak meg középiskolai történelem "
        "és magyar feladatokat, a háttérben keresett tankönyvi kontextusok alapján."
    )
    Story.append(Paragraph(bevezeto, normal_style))
    Story.append(Spacer(1, 15))

    # Értékelési szempontok beemelése
    Story.append(Paragraph("Az LLM-as-a-Judge kiértékelés szempontjai (1-5 skálán):", sub_title_style))
    szempontok = [
        "<b>1. Faithfulness (Tényhűség és Forrásfegyelem):</b> Kizárólag a RAG kontextusból dolgozik-e, leleplezi-e a hiányzó adatokat és anakronizmusokat, vagy hallucinál és külső tudást használ.",
        "<b>2. Role Adherence (Perszóna és Szabálykövetés):</b> Szigorúan betartja-e a szerep specifikus negatív és pozitív instrukcióit (pl. szókratészi rávezetés tiltott direkt válaszadással, forrásmegjelölések alkalmazása, modern analógiák beépítése).",
        "<b>3. Pedagogical Utility (Didaktikai Minőség):</b> Korosztályhoz illeszkedő-e a nyelvezet, világos-e a fogalmi struktúra, és könnyen tanulható-e a magyarázat."
    ]
    for szempont in szempontok:
        Story.append(Paragraph(szempont, normal_style))
    Story.append(Spacer(1, 15))

    # System Prompts kiírása
    Story.append(Paragraph("Az alkalmazott System Prompt-ok (Perszónák):", sub_title_style))
    for p_name, p_text in PERSONAS.items():
        Story.append(Paragraph(f"<b>{p_name}</b>", normal_style))
        Story.append(Paragraph(f"<i>{p_text}</i>", normal_style))
        Story.append(Spacer(1, 10))
        
    Story.append(Spacer(1, 10))
    
    # Tesztkérdések kiírása
    Story.append(Paragraph("A diákok feltett tesztkérdései (User Prompts):", sub_title_style))
    for q_name, q_text in TEST_QUESTIONS.items():
        Story.append(Paragraph(f"<b>{q_name}</b>", normal_style))
        Story.append(Paragraph(f"<i>{q_text}</i>", normal_style))
        Story.append(Spacer(1, 10))

    # Oldaltörés a tényleges eredmények előtt
    Story.append(PageBreak())
    
    Story.append(Paragraph("Részletes Teszteredmények", title_style))
    Story.append(Spacer(1, 10))

    for res in results_list:
        blokk = []
        blokk.append(Paragraph(f"Modell: {res['Modell']} | Idő: {res['Idő']}", sub_title_style))
        blokk.append(Paragraph(f"Perszóna: {res['Persona']}", normal_style))
        blokk.append(Paragraph(f"Kérdés: {res['Kerdes']}", normal_style))
        blokk.append(Spacer(1, 10))
        
        blokk.append(Paragraph("A modell válasza:", bold_style))
        safe_valasz = res['Válasz'].replace('\n', '<br/>')
        blokk.append(Paragraph(safe_valasz, normal_style))
        blokk.append(Spacer(1, 10))

        blokk.append(Paragraph("LLM-as-a-Judge (Bírói) Értékelés:", bold_style))
        eval_data = res['Értékelés']
        if "error" in eval_data:
            blokk.append(Paragraph(f"HIBA AZ ÉRTÉKELÉSNÉL: {eval_data['error']}", normal_style))
        else:
            f_score = eval_data.get('faithfulness_score', '-')
            r_score = eval_data.get('role_adherence_score', '-')
            p_score = eval_data.get('pedagogical_score', '-')
            comp = eval_data.get('composite_score', '-')
            
            pontok = f"Összesített: {comp}/5.0 | Hűség: {f_score}/5 | Szerep: {r_score}/5 | Pedagógia: {p_score}/5"
            blokk.append(Paragraph(pontok, score_style))
            blokk.append(Paragraph(f"<b>Hűség indoklás:</b> {eval_data.get('faithfulness_reason', '')}", normal_style))
            blokk.append(Paragraph(f"<b>Szerep indoklás:</b> {eval_data.get('role_adherence_reason', '')}", normal_style))
            blokk.append(Paragraph(f"<b>Pedagógia indoklás:</b> {eval_data.get('pedagogical_reason', '')}", normal_style))
        
        blokk.append(Spacer(1, 20))
        blokk.append(Paragraph("-" * 80, normal_style))
        blokk.append(Spacer(1, 20))
        Story.append(KeepTogether(blokk))

    doc.build(Story)
# =====================================================================
# 7. FELÜLET ÉS NAVIGÁCIÓ

with st.sidebar:
    st.header("Navigáció")
    oldal = st.radio("Válassz modult:", ("Benchmark", "AI Magántanár Chat"))
    st.markdown("---")
    st.header("Rendszervezérlő")
    if st.button("GPU Memória Ürítése"):
        flush_memory()
        st.success("VRAM sikeresen felszabadítva!")

if oldal == "Benchmark":
    st.title("TDK LLM Értékelő Laboratórium")
    st.markdown("Az LLM-as-a-Judge rendszer automatikusan kiértékeli a modellek teljesítményét **3 dimenzióban (Faithfulness, Role Adherence, Pedagogical Utility)**.")
    
    if not index:
        st.error("Hiba: Nem találom az indexelt dokumentumokat.")
        st.stop()

    with st.expander("API és Bíró Beállítások", expanded=False):
        openrouter_api_key = st.text_input("OpenRouter API Kulcs (A Bíróhoz és a Felhős modellekhez):", type="password")
        judge_model = st.selectbox("Bíró Modell (LLM-as-a-Judge):", ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"])

    st.markdown("### A Kísérlet Beállítása")
    selected_persona_name = st.selectbox("1. Szerep / System Prompt:", list(PERSONAS.keys()))
    selected_persona_text = PERSONAS[selected_persona_name]
    st.info(f"**Instrukció:**\n\n_{selected_persona_text}_")
    
    st.markdown("---")
    selected_question_name = st.selectbox("2. Tesztkérdés / User Prompt:", list(TEST_QUESTIONS.keys()))
    kerdes = TEST_QUESTIONS[selected_question_name]
    st.success(f"**Feltett kérdés:**\n\n_{kerdes}_")
    
    st.markdown("### Tesztelendő Modellek")
    col1, col2, col3 = st.columns(3)
    with col1:
        run_gemma = st.checkbox("Gemma 3 (12B) [Lokális]", value=True)
        run_claude = st.checkbox("Claude 3.5 Sonnet [Felhő]", value=True)
    with col2:
        run_racka = st.checkbox("Racka (4B) [Lokális]", value=True)
        run_gpt4o = st.checkbox("GPT-4o [Felhő]", value=False)
    with col3:
        run_gpt_mini = st.checkbox("GPT-4o Mini [Felhő]", value=False)
        run_llama70b = st.checkbox("Llama 3.3 (70B) [Felhő]", value=False)

    if st.button("Kísérlet Futtatása és Értékelés", type="primary"):
        if not openrouter_api_key:
            st.error("Az értékeléshez meg kell adnod az OpenRouter API kulcsot!")
            st.stop()
            
        eredmenyek = []
        qa_prompt_tmpl_str = selected_persona_text + "\n\nKontextus:\n{context_str}\n\nKérdés: {query_str}\nVálasz:"
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        def run_test_for_model(model_name, llm_instance):
            with st.spinner(f"Keresés és generálás: {model_name}..."):
                query_engine = index.as_query_engine(
                    llm=llm_instance, 
                    similarity_top_k=6, 
                    node_postprocessors=[reranker],
                    response_mode="compact"
                )
                query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
                
                start_time = time.time()
                valasz_obj = query_engine.query(kerdes)
                end_time = time.time()
                
                retrieved_chunks = "\n\n".join([node.node.text for node in valasz_obj.source_nodes])
                valasz_szoveg = str(valasz_obj)
                
            with st.spinner(f"Bírói értékelés: {model_name}..."):
                judge_eval = evaluate_with_judge(
                    user_prompt=kerdes, 
                    retrieved_chunks=retrieved_chunks, 
                    system_prompt=selected_persona_text, 
                    model_response=valasz_szoveg, 
                    api_key=openrouter_api_key,
                    judge_model=judge_model
                )
            
            return {
                "Modell": model_name,
                "Idő": f"{end_time - start_time:.2f} mp",
                "Válasz": valasz_szoveg,
                "Források": retrieved_chunks,
                "Értékelés": judge_eval
            }

        if run_gemma:
            flush_memory()
            llm_gemma = Ollama(model="gemma3:12b", request_timeout=360.0)
            eredmenyek.append(run_test_for_model("Gemma 3 (12B)", llm_gemma))
            unload_model("gemma3:12b")
            
        if run_racka:
            flush_memory()
            llm_racka = Ollama(
                model="racka-magantanar", 
                request_timeout=600.0,
                additional_kwargs={"num_predict": 800, "repeat_penalty": 1.2}
            )
            eredmenyek.append(run_test_for_model("Racka (4B)", llm_racka))
            unload_model("racka-magantanar")

        cloud_models = []
        if run_gpt_mini: cloud_models.append(("GPT-4o Mini", "openai/gpt-4o-mini"))
        if run_claude: cloud_models.append(("Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet"))
        if run_gpt4o: cloud_models.append(("GPT-4o", "openai/gpt-4o"))
        if run_llama70b: cloud_models.append(("Llama 3.3 (70B)", "meta-llama/llama-3.3-70b-instruct"))

        for m_name, m_id in cloud_models:
            llm_cloud = OpenAILike(
                model=m_id, 
                api_key=openrouter_api_key, 
                api_base="https://openrouter.ai/api/v1", 
                is_chat_model=True,
                request_timeout=120.0,
                default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "TDK-Benchmark"}
            )
            eredmenyek.append(run_test_for_model(m_name, llm_cloud))

        if eredmenyek:
            st.success("A kísérlet és az értékelés sikeresen lefutott!")
            for eredmeny in eredmenyek:
                with st.expander(f"{eredmeny['Modell']} - Reakcióidő: {eredmeny['Idő']}", expanded=True):
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.markdown("**A Modell Válasza:**")
                        st.write(eredmeny['Válasz'])
                        st.markdown("---")
                        st.markdown("*A modell által felhasznált háttéranyag (RAG)*")
                        st.caption(eredmeny['Források'][:500] + "..." if len(eredmeny['Források']) > 500 else eredmeny['Források'])
                    with col_b:
                        eval_data = eredmeny['Értékelés']
                        if "error" in eval_data:
                            st.error(f"Hiba az értékelésnél: {eval_data['error']}")
                        else:
                            f_score = eval_data.get('faithfulness_score', '-')
                            r_score = eval_data.get('role_adherence_score', '-')
                            p_score = eval_data.get('pedagogical_score', '-')
                            comp = eval_data.get('composite_score', '-')
                            st.metric("Összesített Pontszám", f"{comp} / 5.0")
                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric("Hűség (RAG)", f"{f_score}/5")
                            m_col2.metric("Szerepkövetés", f"{r_score}/5")
                            m_col3.metric("Pedagógia", f"{p_score}/5")
                            st.markdown("---")
                            st.markdown(f"**Tényhűség értékelése:**\n_{eval_data.get('faithfulness_reason', '')}_")
                            st.markdown(f"**Szerep/Szabálykövetés értékelése:**\n_{eval_data.get('role_adherence_reason', '')}_")
                            st.markdown(f"**Didaktikai minőség:**\n_{eval_data.get('pedagogical_reason', '')}_")

    # =====================================================================
    # 8. Tömeges tesztek és Monitorozás

    if "show_plot" not in st.session_state:
        st.session_state.show_plot = False
    if "model_intervals" not in st.session_state:
        st.session_state.model_intervals = []

    st.markdown("---")
    st.markdown("### Mindent Futtat és PDF-be Ment")
    
    if st.button("Tömeges Teszt Indítása", type="secondary"):
        if not openrouter_api_key:
            st.error("Add meg az OpenRouter API kulcsot a Bíróhoz és a felhős modellekhez!")
            st.stop()

        st.session_state.show_plot = False
        st.session_state.model_intervals = []
            
        eredmenyek_pdfhez = []
        ossz_teszt = len(PERSONAS) * len(TEST_QUESTIONS)
        aktiv_modellek = sum([run_gemma, run_racka, run_gpt_mini, run_claude, run_gpt4o, run_llama70b])
        
        if aktiv_modellek == 0:
            st.error("Jelölj be legalább egy modellt a fenti listából!")
            st.stop()

        total_steps = ossz_teszt * aktiv_modellek
        my_bar = st.progress(0, text="Felkészülés a tömeges futtatásra...")
        
        def futtat_egy_tesztet_biztonsagosan(modell_nev, llm_instance, p_name, p_text, q_name, q_text, is_cloud=False):
            if is_cloud:
                time.sleep(1.5)
                
            qa_tmpl = PromptTemplate(p_text + "\n\nKontextus:\n{context_str}\n\nKérdés: {query_str}\nVálasz:")
            try:
                query_eng = index.as_query_engine(
                    llm=llm_instance, 
                    similarity_top_k=6,
                    node_postprocessors=[reranker],
                    response_mode="compact"
                )
                query_eng.update_prompts({"response_synthesizer:text_qa_template": qa_tmpl})
                
                s_time = time.time()
                v_obj = query_eng.query(q_text)
                e_time = time.time()
                
                chunkok = "\n\n".join([node.node.text for node in v_obj.source_nodes])
                v_szoveg = str(v_obj)
                
                j_eval = evaluate_with_judge(q_text, chunkok, p_text, v_szoveg, openrouter_api_key, judge_model)
                
                return {
                    "Modell": modell_nev,
                    "Persona": p_name,
                    "Kerdes": q_name,
                    "Idő": f"{e_time - s_time:.2f} mp",
                    "Válasz": v_szoveg,
                    "Források": chunkok,
                    "Értékelés": j_eval
                }
            except Exception as e:
                return {
                    "Modell": modell_nev,
                    "Persona": p_name,
                    "Kerdes": q_name,
                    "Idő": "HIBA mp",
                    "Válasz": f"Hiba történt a futtatás közben: {str(e)}",
                    "Források": "N/A",
                    "Értékelés": {"error": str(e)}
                }

        szamlalo = 0
        monitor_stop_event = threading.Event()
        monitor_thread = threading.Thread(target=hardware_monitor, args=(monitor_stop_event, "TDK_hardware_log.csv"))
        monitor_thread.start()

        try:
            if run_gemma:
                start_t = datetime.now().strftime("%H:%M:%S")
                flush_memory()
                llm_gemma = Ollama(model="gemma3:12b", request_timeout=360.0)
                for p_name, p_text in PERSONAS.items():
                    for q_name, q_text in TEST_QUESTIONS.items():
                        my_bar.progress(min(1.0, szamlalo / total_steps), text=f"Gemma 3 (12B) dolgozik... ({szamlalo + 1}/{total_steps})")
                        res = futtat_egy_tesztet_biztonsagosan("Gemma 3 (12B)", llm_gemma, p_name, p_text, q_name, q_text, is_cloud=False)
                        eredmenyek_pdfhez.append(res)
                        szamlalo += 1
                unload_model("gemma3:12b")
                st.session_state.model_intervals.append({"model": "Gemma 3 (12B)", "start": start_t, "end": datetime.now().strftime("%H:%M:%S")})

            if run_racka:
                start_t = datetime.now().strftime("%H:%M:%S")
                flush_memory()
                llm_racka = Ollama(
                    model="racka-magantanar", 
                    request_timeout=600.0, 
                    additional_kwargs={"num_predict": 800, "repeat_penalty": 1.2}
                )
                for p_name, p_text in PERSONAS.items():
                    for q_name, q_text in TEST_QUESTIONS.items():
                        my_bar.progress(min(1.0, szamlalo / total_steps), text=f"Racka (4B) dolgozik... ({szamlalo + 1}/{total_steps})")
                        res = futtat_egy_tesztet_biztonsagosan("Racka (4B)", llm_racka, p_name, p_text, q_name, q_text, is_cloud=False)
                        eredmenyek_pdfhez.append(res)
                        szamlalo += 1
                unload_model("racka-magantanar")
                st.session_state.model_intervals.append({"model": "Racka (4B)", "start": start_t, "end": datetime.now().strftime("%H:%M:%S")})

            cloud_models_bulk = []
            if run_gpt_mini: cloud_models_bulk.append(("GPT-4o Mini", "openai/gpt-4o-mini"))
            if run_claude: cloud_models_bulk.append(("Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet"))
            if run_gpt4o: cloud_models_bulk.append(("GPT-4o", "openai/gpt-4o"))
            if run_llama70b: cloud_models_bulk.append(("Llama 3.3 (70B)", "meta-llama/llama-3.3-70b-instruct"))

            for m_name, m_id in cloud_models_bulk:
                start_t = datetime.now().strftime("%H:%M:%S")
                llm_cloud = OpenAILike(
                    model=m_id, 
                    api_key=openrouter_api_key, 
                    api_base="https://openrouter.ai/api/v1", 
                    is_chat_model=True,
                    request_timeout=120.0,
                    default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "TDK-Benchmark"}
                )
                for p_name, p_text in PERSONAS.items():
                    for q_name, q_text in TEST_QUESTIONS.items():
                        my_bar.progress(min(1.0, szamlalo / total_steps), text=f"{m_name} dolgozik... ({szamlalo + 1}/{total_steps})")
                        res = futtat_egy_tesztet_biztonsagosan(m_name, llm_cloud, p_name, p_text, q_name, q_text, is_cloud=True)
                        eredmenyek_pdfhez.append(res)
                        szamlalo += 1
                st.session_state.model_intervals.append({"model": m_name, "start": start_t, "end": datetime.now().strftime("%H:%M:%S")})

            my_bar.progress(1.0, text="Kész! PDF generálása folyamatban...")
            fajlnev = "TDK_Teljes_Benchmark_Jelentes.pdf"
            generate_tdk_pdf_report(eredmenyek_pdfhez, fajlnev)
            st.success(f"A teljes mérés lefutott ({len(eredmenyek_pdfhez)} teszt). Fájl: **{fajlnev}**")
            st.session_state.show_plot = True

        finally:
            monitor_stop_event.set()
            monitor_thread.join()

    # --- GRAFIKON ÉS LETÖLTÉS MEGJELENÍTÉSE ---
    if st.session_state.show_plot:
        st.markdown("### Hardver terhelés és Modellek futási ideje")
        try:
            df = pd.read_csv("TDK_hardware_log.csv")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=df['Idopont'], y=df['VRAM_GB'], mode='lines', name='VRAM (GB)', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df['Idopont'], y=df['GPU_szazalek'], mode='lines', name='GPU Terhelés (%)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['Idopont'], y=df['RAM_GB'], mode='lines', name='Rendszer RAM (GB)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Idopont'], y=df['CPU_szazalek'], mode='lines', name='CPU Terhelés (%)', line=dict(color='green')))
            
            szinek = ["rgba(0, 0, 255, 0.1)", "rgba(0, 255, 0, 0.1)", "rgba(255, 0, 0, 0.1)", "rgba(255, 255, 0, 0.1)", "rgba(255, 0, 255, 0.1)", "rgba(0, 255, 255, 0.1)"]
            
            for idx, interval in enumerate(st.session_state.model_intervals):
                szin = szinek[idx % len(szinek)]
                fig.add_vrect(
                    x0=interval["start"], x1=interval["end"],
                    fillcolor=szin, opacity=1,
                    layer="below", line_width=1, line_dash="dash",
                    annotation_text=interval["model"], annotation_position="top left"
                )

            fig.update_layout(height=500, xaxis_title="Idő", yaxis_title="Értékek", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            with open("TDK_Teljes_Benchmark_Jelentes.pdf", "rb") as pdf_file:
                st.download_button(label="Töltsd le a frissített PDF-et", data=pdf_file, file_name="TDK_Teljes_Benchmark_Jelentes.pdf", mime="application/pdf")
                
        except Exception as e:
            st.error(f"Nem sikerült betölteni a hardver logot a grafikonhoz: {e}")

# =====================================================================
# 9. CHAT OLDAL

elif oldal == "AI Magántanár Chat":
    st.title("AI Magántanár (Ágens)")
    if not index:
        st.error("Nincs betöltve az index! Kérlek várj, amíg a rendszer feldolgozza a dokumentumokat.")
        st.stop()

    with st.sidebar:
        st.markdown("### Motor Beállítása (Chat)")
        llm_choice = st.radio(
            "Válaszd ki a modellt:",
            ("Lokális - Gemma 3 (12B)", "Lokális - Racka (4B)", "Felhős - OpenRouter")
        )
        openrouter_api_key = ""
        openrouter_model = ""
        
        if llm_choice == "Felhős - OpenRouter":
            openrouter_api_key = st.text_input("OpenRouter API Kulcs:", type="password")
            or_model_preset = st.selectbox(
                "Válassz felhős modellt:",
                (
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-2.5-flash",
                    "openai/gpt-4o",
                    "meta-llama/llama-3.3-70b-instruct",
                    "Egyéb (Kézi megadás)"
                )
            )
            if or_model_preset == "Egyéb (Kézi megadás)":
                openrouter_model = st.text_input("Írd be a modell pontos azonosítóját:")
            else:
                openrouter_model = or_model_preset
                
        st.markdown("---")
        if st.button("Beszélgetés Törlése"):
            st.session_state.messages = []
            if "agent_ctx" in st.session_state:
                del st.session_state.agent_ctx
            st.rerun()

    current_llm = None
    if llm_choice == "Lokális - Gemma 3 (12B)":
        flush_memory() 
        current_llm = Ollama(model="gemma3:12b", request_timeout=360.0)
    elif llm_choice == "Lokális - Racka (4B)":
        flush_memory()
        current_llm = Ollama(model="racka-magantanar", request_timeout=360.0)
    elif llm_choice == "Felhős - OpenRouter":
        if not openrouter_api_key or not openrouter_model:
            st.warning("Kérlek, add meg az OpenRouter API kulcsot és válassz modellt!")
            st.stop()
        current_llm = OpenAILike(
            model=openrouter_model, 
            api_key=openrouter_api_key, 
            api_base="https://openrouter.ai/api/v1", 
            is_chat_model=True,
            request_timeout=120.0,
            default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "TDK-Benchmark"}
        )

    query_engine = index.as_query_engine(
        llm=current_llm, 
        similarity_top_k=6,
        node_postprocessors=[reranker],
        response_mode="compact"
    )
    tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="tankonyv_kereso",
                description="Használd ezt a szerszámot minden ténybeli, irodalmi vagy történelmi kérdésnél!"
            )
        )
    ]
    
    system_prompt = (
        "Te egy magyar bölcsészprofesszor és precíz magántanár vagy. "
        "A feladatod: a kapott tankönyvi részletekből készíts összefoglalót. "
        "Mielőtt válaszolnál, MINDIG kutass a tudásbázisodban a pontos adatokért a 'tankonyv_kereso' szerszámmal! "
        "SZIGORÚ SZABÁLYOK:\n"
        "1. KIZÁRÓLAG helyes, választékos magyar nyelven válaszolj!\n"
        "2. TILOS más nyelvek szavait használni.\n"
        "3. A válaszod legyen strukturált, használj alcímeket és pontokba szedett listákat.\n"
        "4. Ha nincs információ a könyvben, mondd meg őszintén, ne találj ki semmit!\n"
        "5. A végén MINDIG tegyél fel egy ellenőrző kérdést a diáknak!"
    )
    
    agent = ReActAgent(
        tools=tools,
        llm=current_llm,
        system_prompt=system_prompt,
        verbose=True
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_ctx" not in st.session_state:
        st.session_state.agent_ctx = Context(agent)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    async def get_agent_response(msg):
        return await agent.run(user_msg=msg, ctx=st.session_state.agent_ctx)

    if prompt := st.chat_input("Tedd fel a kérdésed a tankönyvvel kapcsolatban..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("A Tanár a válaszon gondolkodik és a könyvben kutat..."):
                try:
                    response = loop.run_until_complete(get_agent_response(prompt))
                    answer_text = str(response.response) if hasattr(response, "response") else str(response)
                    st.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                except Exception as e:
                    st.error(f"Hiba a feldolgozás során: {e}")
                    