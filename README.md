#  EpiSim  AI-Powered Epidemic Disease Spread Modeling System

A production-grade epidemic simulation and prediction platform built with **Python Flask**, **Plotly**, and **Scikit-learn**.

---

##  Table of Contents
1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Setup Instructions (VS Code)](#setup-instructions-vs-code)
5. [Running the Application](#running-the-application)
6. [API Reference](#api-reference)
7. [Mathematical Models](#mathematical-models)

---

##  Features

| Feature | Description |
|     |
| **SIR Model** | Classic 3-compartment epidemic model |
| **SEIR Model** | Extended model with Exposed compartment |
| **Interactive Controls** | Population, β, γ, σ, ν, μ, δ sliders |
| **5 Chart Types** | Compartment, daily cases, deaths, recovery, heatmap |
| **Scenario Comparison** | No intervention vs Lockdown vs Vaccination |
| **AI Prediction** | Random Forest / Linear Regression forecasting |
| **CSV Export** | Export simulation and comparison data |
| **Responsive Dashboard** | Modern dark-theme UI |

---

##  Tech Stack

- **Backend**: Python 3.11+, Flask 3.0, Flask-CORS
- **Math/Science**: NumPy, SciPy (ODE solver), Pandas
- **Machine Learning**: Scikit-learn (Random Forest, Linear Regression)
- **Visualization**: Plotly 5.x
- **Frontend**: Vanilla HTML/CSS/JS + Plotly.js

---

## Structure

```
epidemic_sim/
├── app.py                          # Flask application & API routes
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── generate_sample_data.py         # Sample dataset generator
│
├── models/
│   ├── __init__.py
│   ├── epidemic_models.py          # SIR and SEIR model classes
│   └── ai_predictor.py             # ML prediction module
│
├── utils/
│   ├── __init__.py
│   ├── visualizer.py               # Plotly chart generators
│   └── data_exporter.py            # CSV export utilities
│
├── templates/
│   └── index.html                  # Full-stack single-page dashboard
│
└── data/
    └── sample_epidemic_dataset.csv # Sample historical data
```

---

##  Setup Instructions (VS Code)

### Prerequisites
- Python 3.11 or higher
- VS Code with Python extension

### Step 1 — Clone / Download
```bash
# Download or unzip the project folder
cd epidemic_sim
```

### Step 2 — Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Application
```bash
python app.py
```

### Step 5 — Open in Browser
Navigate to: **http://127.0.0.1:5000**

---

##  API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/` | Dashboard UI |
| `POST` | `/api/simulate` | Run SIR/SEIR simulation |
| `POST` | `/api/predict` | AI infection forecast |
| `POST` | `/api/compare` | Multi-scenario comparison |
| `GET`  | `/api/export/csv` | Download simulation CSV |
| `GET`  | `/api/export/comparison` | Download comparison CSV |
| `GET`  | `/api/sample-dataset` | Download sample dataset |
| `GET`  | `/api/health` | Health check |

### POST /api/simulate — Request Body
```json
{
  "model":  "SIR",
  "N":      1000000,
  "I0":     10,
  "beta":   0.30,
  "gamma":  0.05,
  "sigma":  0.20,
  "nu":     0.0,
  "mu":     0.001,
  "delta":  0.0,
  "days":   365
}
```

---

##  Mathematical Models

### SIR Model
```
dS/dt = -β·S·I/N - ν·S
dI/dt =  β·S·I/N - γ·I - μ·I
dR/dt =  γ·I     + ν·S
dD/dt =  μ·I

R₀ = β·(1-δ) / γ
```

### SEIR Model
```
dS/dt = -β·S·I/N - ν·S
dE/dt =  β·S·I/N - σ·E
dI/dt =  σ·E     - γ·I - μ·I
dR/dt =  γ·I     + ν·S
dD/dt =  μ·I
```

### Parameter Guide
| Symbol | Name | Typical Range |
|---|---|---|
| β | Transmission rate | 0.1 – 1.0 |
| γ | Recovery rate | 0.01 – 0.5 |
| σ | Incubation rate | 0.1 – 0.5 |
| ν | Vaccination rate | 0 – 0.01 |
| μ | Mortality rate | 0 – 0.05 |
| δ | Social distancing | 0 – 0.99 |

---

##  License
Academic project — for educational use.
