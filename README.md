# Mobile App Review Triage System

An open-source, hybrid intelligence platform built with **Python**, **Sklearn**, and **Google Gemini 1.5/2.5 Flash**. This system operates as a unified telemetry dashboard, empowering Product Managers and Developers to instantly classify extreme volumes of unstructured customer feedback, map semantic intent, and isolate critical application defects.

## Core Features

- **Hybrid Intelligence Logic**: Parses textual semantics flawlessly locally via Logistic Regression arrays (TF-IDF N-Grams), while simultaneously querying GenAI dynamically to output strategic Executive Action Plans on exclusively flagged variables.
- **Aggressive Scalability**: Engineered to accept bulk CSV ingress structures seamlessly throttling API requests dynamically sequentially grouping by individual `App Namespaces`—minimizing limit blockades across the Gemini SDK drastically.
- **Decoupled Architecture**: Cleanly separated Flask endpoints map effortlessly to dedicated NLP matrices, ML memory handlers, and contextual LLM retrieval loops.
- **Premium Telemetry UI**: A purely responsive, dense, vertical "nav-pill" viewport-locked dashboard visualizing global pipeline confidence, dynamic dataset rendering boundaries featuring intuitive `Chart.js` distribution graphs, and responsive tables bound by strict ellipsis css logic.

---

## Architectural Map

```bash
MobileAppReviewTriage
 ┣ services
 ┃ ┣ nlp_service.py   # Securely bounds NLTK downloads, exclusions, & lemmatization
 ┃ ┣ ml_service.py    # Retains isolated access to the Logistic Regression tracking
 ┃ ┗ llm_service.py   # Isolates Kaggle metadata and Gemini 2.5 Generation queries
 ┣ templates
 ┃ ┣ index.html       # The Manifesto overview mapping local datasets
 ┃ ┣ analyzer.html    # Drag and Drop bulk CSV processing & internal ML targeting
 ┃ ┗ results.html     # High-Density tracker rendering AI Action Plans independently
 ┣ app.py             # Pristine Flask router importing encapsulated class modules
 ┣ config.py          # Unified config resolving env secrets & file targets
 ┣ data_prep.py       # Developer script extracting & processing massive Kaggle datasets
 ┣ model_builder.py   # Developer script assembling TF-IDF vectors against VADER logic
 ┗ requirements.txt   # Core Python tracking constraints
```

---

## Setup & Installation Guide

**1. Clone the Environment**
```bash
git clone https://github.com/your-repo/MobileAppReviewTriage.git
cd MobileAppReviewTriage
```

**2. Isolate Dependencies**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

**3. Configure Essential Secrets**
Duplicate or initialize your `.env` constraints containing fundamental Kaggle telemetry keys:
```env
# Critical Environment Constraints
SECRET_KEY=super_secure_development_key
GEMINI_API_KEY=your_native_google_ai_studio_api_key_here
```

**4. Procure Required Models & Storage Matrices**
The ML services dictate the instantiation of serialized arrays targeting fundamental vector processing layers. Run the preparation stages dynamically via `pandas`:
```bash
python data_prep.py
python model_builder.py
```
*(Ensure `googleplaystore_user_reviews.csv` and `googleplaystore.csv` are natively secured locally within the root directory before running the scripts above.)*

---

## Running the Server

Trigger the backend gateway locally bridging port 5000:

```bash
python app.py
```
After successful instantiation, navigate directly to **`http://127.0.0.1:5000`** in your browser to begin testing payloads natively!

---

## Tech Stack & Tooling

| Capability       | Bound Framework |
|------------------|----------------|
| **Backend**      | Flask (Python) |
| **Generative AI**| Google Gemini 2.5 SDK |
| **Local Metrics**| Scikit-Learn, NLTK (VADER) |
| **Data Parsing** | Pandas, Numpy  |
| **Frontend UI**  | Bootstrap 5, Jinja2, FontAwesome, Chart.js  |
