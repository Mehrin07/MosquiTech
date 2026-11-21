<img width="1217" height="397" alt="image" src="https://github.com/user-attachments/assets/4e4b7be1-b279-40d7-8a20-f5bd0d1dddf4" /># 🦟 MosquiTech

**AI-Powered Dengue Early Warning & Urban Risk Intelligence**

MosquiTech is an AI-powered urban dengue early-warning and risk intelligence system developed for the **EDU AI Engineering Hackathon 2026 – AI for Urban Dengue Outbreak** challenge.

It forecasts dengue cases at the ward level for 2, 3, and 4 weeks ahead, classifies wards into risk categories, and provides explainable insights to support earlier public-health interventions.

## 1. Problem Statement

Dengue outbreaks create significant public-health challenges in urban areas. Conventional responses often begin after reported cases increase, leaving limited time for preventive action.

Rainfall, temperature, humidity, population density, waterlogging, and sanitation conditions can influence dengue transmission. However, these factors may affect different wards differently.

MosquiTech uses historical dengue data, weather patterns, and ward-level vulnerability indicators to identify potential risks and provide an early-warning window for decision-makers.

## 2. Key Features

- **Multi-week forecasting:** Predicts dengue cases for 2, 3, and 4 weeks ahead.
- **Ward-level risk classification:** Categorizes wards into LOW, MEDIUM, HIGH, and CRITICAL risk levels.
- **Weather feature engineering:** Uses ward-specific rainfall lags, cumulative rainfall, temperature, and humidity features.
- **Interactive dashboard:** Provides overview metrics, forecasts, ward rankings, risk heatmaps, weather information, and model performance.
- **Explainable AI:** Uses Shapley-value-based explanations and feature importance to identify factors contributing to predictions.
- **AI Health Insights Assistant:** Integrates Google Gemini to answer questions using actual model predictions, weather information, historical data, and explanation results.
- **Evidence-based recommendations:** Provides decision-support suggestions based on ward-level risk information.
- **Prediction transparency:** Separates observed historical information from predicted results.

The Gemini assistant explains model outputs; it does not modify the underlying predictions or risk classifications.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Data Visualization | Plotly |
| Model Serialization | Joblib |
| Excel Data Handling | OpenPyXL |
| API Communication | Requests |
| Environment Configuration | python-dotenv |
| AI Model | Google Gemini API |

## 4. Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Internet connection for installing dependencies and using the optional Gemini assistant

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mehrin07/MosquiTech.git
cd MosquiTech
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

```text
http://localhost:8501
```

The trained models are included in the repository, so model training is not normally required to launch the dashboard.

### Optional: Enable the AI Health Insights Assistant

MosquiTech's forecasting dashboard can operate without a Gemini API key. To enable the AI assistant, configure the following environment variables.

Create a `.env` file in the project root using the example configuration:

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

Restart the Streamlit application after configuring the environment variables.

**Security:** Never commit `.env` or expose API keys in source code, screenshots, or public repositories. The `.env` file is excluded through `.gitignore`.

### Optional: Train the Models Again

To reproduce model training and generate the associated reports:

```bash
python -m src.train_model
```

## 5. AI Tools and External API Disclosure

The following tools and services are used in the development and runtime of MosquiTech.

| Tool / API | Purpose |
|---|---|
| ChatGPT (OpenAI) | Development assistance, code explanations, debugging, and documentation support. |
| OpenAI Codex | Code generation, code modifications, and coding assistance. |
| Claude (Anthropic) | AI-assisted development, code review, and documentation support. |
| Google Gemini | Development assistance, debugging, and technical guidance. |
| Google Gemini API | Runtime AI assistant for generating contextual health insights based on MosquiTech's model outputs and supporting evidence. |

### Gemini API Details

- **Provider:** Google
- **API:** Gemini API (REST)
- **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- **Configured Model:** `gemini-3.1-flash-lite`
- **Configuration:** `GEMINI_API_KEY` and `GEMINI_MODEL` environment variables.
- **Purpose:** Answering user questions about dengue forecasts, ward-level risks, historical trends, weather conditions, and model explanations.

### AI Usage and Safeguards

The AI assistant receives relevant contextual information from the application's existing data and prediction pipeline.

The system is designed to:

- Use actual model predictions and supporting evidence.
- Distinguish observed historical data from predicted values.
- Avoid changing or overriding the machine-learning model's risk classifications.
- Avoid inventing unavailable predictions or unsupported numerical claims.
- Present recommendations as decision support rather than medical advice.
- Handle API errors and network failures without disabling the main forecasting dashboard.

The AI assistant is optional. Core forecasting and dashboard functionality do not depend on the Gemini API.

## 6. Attributions

- The dataset used in this project was provided for the EDU AI Engineering Hackathon 2026.
- The project uses open-source Python libraries and frameworks listed in `requirements.txt`.
- Any external code snippets, assets, or additional resources used by the team should be acknowledged here with their original sources.

## 7. Demo

**GitHub Repository:**  
https://github.com/Mehrin07/MosquiTech

**Application:**  
Run locally using the installation instructions above.

## 8. Team

**Team Name: Aethrix**

| Team Member | GitHub | Role |
|---|---|---|
| Mehrin Ferdous Rafin | [Mehrin07](https://github.com/Mehrin07) | Dashboard Development and AI Assistant Integration |
| Tasnim Tabassum | [Tasnim1502](https://github.com/Tasnim1502) | Data Processing and Feature Engineering |
| Mafruha Tajrian Safa | [tajriansafa](https://github.com/tajriansafa) | Machine Learning, Forecasting, and Model Evaluation |

## 9. Project Structure

```text
MosquiTech/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── ARCHITECTURE.md
│
├── data/
│   └── hackathon_dataset.xlsx
│
├── models/
│   ├── model_h2.joblib
│   ├── model_h3.joblib
│   ├── model_h4.joblib
│   └── metadata.json
│
├── reports/
│   ├── evaluation_report.md
│   ├── risk_validation.md
│   ├── data_quality.json
│   ├── model_comparison.csv
│   └── other evaluation reports
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── prediction.py
│   ├── risk_classification.py
│   ├── explain.py
│   ├── intelligence.py
│   ├── llm_client.py
│   ├── validation.py
│   ├── recommendations.py
│   ├── evaluation.py
│   └── report.py
│
└── tests/
    ├── test_pipeline.py
    ├── test_ai_modules.py
    └── stub_run_app.py
```

## 10. Disclaimer

MosquiTech is a prototype designed for dengue early-warning and urban risk intelligence. Its forecasts and recommendations are intended to support decision-making and are not a substitute for professional medical advice or official public-health surveillance.
