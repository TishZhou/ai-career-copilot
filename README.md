# AI Job Assistant

A web application for job searching, resume–job matching, and intelligent H1B sponsorship filtering—designed for international job seekers.

---

## Overview

The system combines **real-time job scraping**, **H1B-sponsorship intelligence**, and **RAG-powered resume analysis** to help users find relevant roles and receive personalized job-fit insights.

---

## System Components

### 1. Web Scraper — Real-Time Job Collector

The scraper in `final_with_filter.py` uses the **jobspy** library to fetch job postings from:

- LinkedIn  
- Indeed  
- Glassdoor  
- Google Jobs  

When a user enters **Job Title**, **Location**, and **Job Type**, the scraper issues parallel requests and aggregates all matching public job listings.

An intelligent **H1B Sponsorship Filter** analyzes each job description by detecting:

- Positive indicators: `h1b`, `visa`, `sponsorship`  
- Negative indicators: `no sponsorship`, `not eligible for visa`, etc.

Only roles that are likely to support international candidates are shown.

---

### 2. RAG (Retrieval-Augmented Generation) — Resume-Aware AI Assistant

RAG enables the system to generate answers grounded in the user’s actual resume.

**Process:**

1. **Resume Extraction**  
   - PDFs processed via PyPDF2  
   - `.txt` files read directly  
   - All text content extracted

2. **Embedding & Vector Indexing**  
   - Resume text converted into embeddings using `OpenAIEmbeddings`  
   - Embedded vectors stored in a **FAISS** index for similarity search

3. **Context Retrieval**  
   - On each user question (e.g., “Am I qualified for this job?”)  
   - The retriever selects the most relevant resume segments

4. **LLM Response Generation**  
   - The system passes the retrieved context + user query to **gpt-4o**  
   - Produces personalized, context-aware responses

RAG essentially works as an **AI analyst** that grounds every answer in resume facts.

---

## How to Run

To avoid API key leakage, run the application using:

```bash
OPENAI_API_KEY="<your-key>" streamlit run final.py