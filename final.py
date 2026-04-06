import os
import json
import time
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# =============================
# 0. Global Config & Setup
# =============================

load_dotenv()

# Streamlit Page Config (Must be the first command)
st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .suggestion-card {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

# API Key Check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please configure it in your .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma Vector Store (Persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
kb_collection = chroma_client.get_or_create_collection(
    name="career_kb_v3_english",  # Changed name to ensure fresh DB
    metadata={"hnsw:space": "cosine"}
)

# =============================
# 1. Basic Tools: Embedding & LLM
# =============================

def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    # Using text-embedding-3-small for cost-efficiency
    resp = client.embeddings.create(
        model="openai.text-embedding-3-small",
        input=[text]
    )
    return resp.data[0].embedding

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "openai.gpt-4o-mini", 
    temperature: float = 0.3,
    json_mode: bool = False
) -> str:
    response_format = {"type": "json_object"} if json_mode else {"type": "text"}
    
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=response_format
    )
    return resp.choices[0].message.content

# =============================
# 2. RAG Core: Seed Knowledge & Retrieval
# =============================

def seed_knowledge_base():
    """
    Injects professional career advice into the RAG system.
    """
    if kb_collection.count() > 5:
        return

    knowledge_data = [
        "【STAR Method】When describing experiences, always use the Situation, Task, Action, Result structure. Quantify results whenever possible (e.g., 'Improved efficiency by 20%').",
        "【Action Verbs】Avoid passive words like 'Responsible for' or 'Helped'. Use strong action verbs such as 'Architected', 'Deployed', 'Optimized', 'Spearheaded', or 'Engineered'.",
        "【Formatting】Ensure the resume is scannable (ATS friendly). Key skills should be listed clearly at the top or bottom. Stick to standard fonts and avoid complex graphics for technical roles.",
        "【Quantifying Impact】Instead of saying 'Fixed bugs', say 'Resolved critical bugs that reduced system downtime by 99.9% and saved $50k in potential losses'.",
        "【Job Matching】If a job requires specific tools (e.g., Python, SQL) and you lack one, highlight your ability to learn quickly by referencing similar tools you have mastered in the past."
    ]

    ids = [f"kb_en_{i}" for i in range(len(knowledge_data))]
    embeddings = [get_embedding(txt) for txt in knowledge_data]
    metadatas = [{"source": "Expert_Career_Guide_v1"} for _ in knowledge_data]

    kb_collection.add(
        ids=ids,
        documents=knowledge_data,
        metadatas=metadatas,
        embeddings=embeddings
    )

def rag_retrieve(query: str, top_k: int = 3) -> str:
    """Retrieves relevant career advice."""
    results = kb_collection.query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k
    )
    
    if not results or not results['documents']:
        return ""

    context_str = ""
    for idx, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][idx].get('source', 'unknown')
        context_str += f"- {doc}\n"
    return context_str

# =============================
# 3. Logic: Parsing, Scoring & Agent
# =============================

def parse_resume(text: str) -> Dict:
    """Parses raw resume text into structured JSON."""
    sys_p = "You are an expert resume parser. You must output strictly valid JSON."
    usr_p = f"""
    Extract the following fields from the resume text below:
    - name (string)
    - summary (string, max 50 words)
    - skills (list of strings)
    
    Resume Text:
    {text[:3000]}
    """
    try:
        res = call_llm(sys_p, usr_p, json_mode=True)
        return json.loads(res)
    except:
        return {"name": "Candidate", "summary": "Parse failed", "skills": []}

def calculate_match_score(resume_text: str, jd_text: str) -> Dict:
    """
    Calculates fit score, keywords, and improvement suggestions.
    """
    sys_p = "You are a Senior Technical Recruiter AI. Analyze the fit between a Resume and a Job Description."
    usr_p = f"""
    Job Description:
    {jd_text}
    
    Resume:
    {resume_text[:3000]}
    
    Task:
    1. Assign a Match Score from 0 to 100.
    2. Identify 3-5 Missing Keywords (skills in JD but not in Resume).
    3. Identify 3-5 Matching Keywords.
    4. Provide a 1-sentence concise Verdict.
    5. Provide 3 specific, actionable Improvement Suggestions to increase the match score.
    
    Output strictly valid JSON with this schema:
    {{
        "score": 85,
        "missing_keywords": ["React", "AWS"],
        "matching_keywords": ["Python", "SQL"],
        "verdict": "Strong candidate but lacks specific cloud experience.",
        "improvement_suggestions": [
            "Your resume lacks 'AWS'. Consider adding a project that uses EC2.",
            "Move your 'Skills' section to the top.",
            "Quantify the impact in your Data Analyst role."
        ]
    }}
    """
    try:
        res = call_llm(sys_p, usr_p, json_mode=True)
        return json.loads(res)
    except:
        return {
            "score": 0, 
            "missing_keywords": [], 
            "matching_keywords": [], 
            "verdict": "Error analyzing match.",
            "improvement_suggestions": ["Error generating suggestions."]
        }

def optimize_bullet_point(bullet: str, jd_text: str) -> Dict:
    """Rewrites a single bullet point using RAG knowledge (STAR method)."""
    rag_advice = rag_retrieve("How to write good bullet points STAR method action verbs")
    
    sys_p = "You are an expert Resume Editor. You specialize in the STAR method."
    usr_p = f"""
    ORIGINAL BULLET: "{bullet}"
    TARGET JOB CONTEXT: {jd_text[:1000]}
    EXPERT GUIDELINES: {rag_advice}
    
    TASK: Rewrite the bullet point in 3 ways.
    Output ONLY JSON:
    {{
        "star_version": "Situation -> Task -> Action -> Result string",
        "quantified_version": "Version with numbers/metrics string",
        "aligned_version": "Version with JD keywords string"
    }}
    """
    try:
        res = call_llm(sys_p, usr_p, json_mode=True)
        return json.loads(res)
    except:
        return {"star_version": "Error", "quantified_version": "Error", "aligned_version": "Error"}

def agent_response(user_input: str, history: list, context_data: dict):
    """Chatbot logic with RAG context."""
    rag_context = rag_retrieve(user_input)
    
    system_prompt = f"""
    You are an AI Career Coach named 'Career Copilot'.
    USER CONTEXT: Resume Summary: {context_data.get('resume_summary', 'N/A')}, Match Score: {context_data.get('match_score', 'N/A')}
    RETRIEVED KNOWLEDGE (RAG): {rag_context}
    INSTRUCTIONS: Answer professionally. Use the RAG knowledge if relevant. Keep it concise.
    """

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    resp = client.chat.completions.create(
        model="openai.gpt-4o-mini",
        messages=messages,
        temperature=0.5
    )
    return resp.choices[0].message.content, rag_context

# =============================
# 4. Streamlit UI (Main)
# =============================

def main():
    # Initialize Session State
    if "messages" not in st.session_state: st.session_state.messages = []
    if "resume_text" not in st.session_state: st.session_state.resume_text = ""
    if "match_analysis" not in st.session_state: st.session_state.match_analysis = None
    if "parsed_resume" not in st.session_state: st.session_state.parsed_resume = None

    # Seed Knowledge Base
    seed_knowledge_base()

    # --- Sidebar ---
    with st.sidebar:
        st.header("📂 Setup")
        uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type="pdf")
        if uploaded_file:
            reader = PdfReader(uploaded_file)
            st.session_state.resume_text = "".join([page.extract_text() for page in reader.pages])
            if not st.session_state.parsed_resume:
                with st.spinner("Parsing Structure..."):
                    st.session_state.parsed_resume = parse_resume(st.session_state.resume_text)
                st.success("Resume Parsed!")

        jd_input = st.text_area("2. Job Description", height=150, placeholder="Paste JD here...")
        
        if st.button("🚀 Analyze Match", type="primary"):
            if st.session_state.resume_text and jd_input:
                with st.spinner("Analyzing Match & Suggestions..."):
                    st.session_state.match_analysis = calculate_match_score(st.session_state.resume_text, jd_input)
                    st.session_state.messages.append({"role": "assistant", "content": f"Analysis complete! Your score is {st.session_state.match_analysis['score']}/100."})
            else:
                st.warning("Please upload a resume and paste a JD.")

        if st.button("Clear History"):
            st.session_state.messages = []
            st.rerun()

    # --- Main Page ---
    st.title("AI Career Copilot")
    st.caption("Powered by RAG, GPT-4o, and Vector Search")

    # 1. Dashboard Section
    if st.session_state.match_analysis:
        analysis = st.session_state.match_analysis
        
        # Row 1: Key Metrics
        c1, c2, c3 = st.columns([1, 2, 2])
        c1.metric("Match Score", f"{analysis['score']}%")
        
        with c2:
            st.markdown("##### ✅ Matching Skills")
            tags = [f"`{k}`" for k in analysis.get('matching_keywords', [])]
            st.markdown(" ".join(tags) if tags else "None")
            
        with c3:
            st.markdown("##### ⚠️ Missing Skills")
            tags = [f"`{k}`" for k in analysis.get('missing_keywords', [])]
            st.markdown(" ".join(tags) if tags else "None")

        st.info(f"**Verdict:** {analysis.get('verdict', 'N/A')}")
        
        # Row 2: Improvement Suggestions (The new feature)
        st.subheader("💡 Actionable Improvement Suggestions")
        suggestions = analysis.get("improvement_suggestions", [])
        if suggestions:
            for idx, tip in enumerate(suggestions, 1):
                st.warning(f"**{idx}.** {tip}")
        else:
            st.write("No specific suggestions found.")
        
        st.divider()

    # 2. Bullet Point Polisher Section
    with st.expander("AI Bullet Point Polisher", expanded=False):
        st.caption("Paste a weak bullet point below to see STAR-method improvements.")
        b_col1, b_col2 = st.columns([1, 1])
        
        with b_col1:
            bullet_input = st.text_area("Original Bullet Point", height=100, placeholder="e.g. Worked on backend code.")
            if st.button("Polish Bullet"):
                if bullet_input and jd_input:
                    with st.spinner("Polishing..."):
                        st.session_state['optimized_bullet'] = optimize_bullet_point(bullet_input, jd_input)
                else:
                    st.error("Need Bullet Point AND Job Description.")
        
        with b_col2:
            if 'optimized_bullet' in st.session_state:
                res = st.session_state['optimized_bullet']
                st.markdown(f"**🔹 STAR Version:**\n> {res.get('star_version')}")
                st.markdown(f"**🔹 Quantified:**\n> {res.get('quantified_version')}")
                st.markdown(f"**🔹 JD Aligned:**\n> {res.get('aligned_version')}")

    st.divider()

    # 3. Chat Interface
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Ask about your resume strategies..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        context_data = {
            "resume_summary": st.session_state.parsed_resume.get("summary", "") if st.session_state.parsed_resume else "",
            "match_score": st.session_state.match_analysis.get("score", "") if st.session_state.match_analysis else ""
        }
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp, rag_info = agent_response(prompt, st.session_state.messages, context_data)
                st.markdown(resp)
                if rag_info:
                    with st.expander("RAG Source"):
                        st.markdown(rag_info)
        
        st.session_state.messages.append({"role": "assistant", "content": resp})

if __name__ == "__main__":
    main()