# 📈 Stock Price Prediction Using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/YFinance-Stock%20Data-111111?style=for-the-badge&logo=yahoo&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-LSTM-8A2BE2?style=for-the-badge" />
</p>

<p align="center">
  <b>An intelligent and secure web-based stock price prediction system using LSTM deep learning.</b>
</p>

## Project Overview

**Stock Price Prediction Using Machine Learning** is a secure, interactive web application designed to predict the **next trading day's closing price** of a selected stock using a **Long Short-Term Memory (LSTM)** deep learning model.

The application retrieves historical and near-real-time stock market data through **Yahoo Finance using the `yfinance` Python library**, preprocesses the data using **MinMaxScaler**, and uses an LSTM model to identify sequential patterns in historical stock prices.

The predicted price is compared with the latest stock price to generate an automated **BUY, SELL, or HOLD** recommendation. The application also provides graphical analysis such as moving averages and actual-vs-predicted price comparisons.

---

## Objectives

* Develop a real-time stock price prediction system using an **LSTM neural network**.
* Retrieve historical and intraday stock data using **Yahoo Finance and yfinance**.
* Predict the **next trading day's closing price**.
* Preprocess stock data using **MinMaxScaler normalization**.
* Compare predicted prices with actual market prices.
* Evaluate model performance using **Root Mean Square Error (RMSE)**.
* Generate automated **BUY, SELL, or HOLD** recommendations.
* Provide stock trend visualizations and moving-average charts.
* Develop a simple and interactive browser-based interface using **Streamlit**.
* Implement multi-factor authentication using password login, CAPTCHA, OTP, and Google OAuth.

---

## Key Features
### Stock Price Prediction

Predicts the next trading day's closing price using an LSTM deep learning model trained on historical stock data.

### Real-Time & Historical Data

Fetches historical and intraday market information through Yahoo Finance using `yfinance`.

### LSTM Deep Learning Model

Uses stacked LSTM layers to learn temporal dependencies and sequential patterns in stock price data.

### Moving Average Analysis

Displays Simple Moving Average (SMA) charts to help users understand historical price trends.

### BUY / SELL / HOLD Recommendation

The system compares the predicted next-day price with the latest available price and generates a recommendation based on the percentage change.

* **BUY** → Predicted price indicates a significant upward movement
* **SELL** → Predicted price indicates a significant downward movement

### Actual vs Predicted Visualization

Provides graphical comparison between actual stock prices and model predictions.

### Secure Authentication

The application includes multiple security mechanisms:

* Username and password authentication
* SHA-256 password hashing
* CAPTCHA verification
* Six-digit email OTP verification
* Gmail SMTP integration
* Google OAuth 2.0 login

The OTP is time-limited and the authentication workflow is designed to reduce unauthorized access and automated attacks.

### Interactive Web Application

Built using Streamlit, allowing users to interact with the prediction system directly through a web browser.

---

## System Architecture

The application follows a **three-tier architecture**:

```text
                 ┌─────────────────────────┐
                 │       USER / BROWSER     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    PRESENTATION TIER    │
                 │       Streamlit UI      │
                 │                         │
                 │ • Login / Sign-up       │
                 │ • CAPTCHA               │
                 │ • OTP Verification      │
                 │ • Stock Input           │
                 │ • Charts & Predictions  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       LOGIC TIER        │
                 │      Python Modules     │
                 │                         │
                 │ • Authentication        │
                 │ • Data Processing       │
                 │ • LSTM Inference        │
                 │ • Recommendation        │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │        DATA TIER        │
                 │                         │
                 │ • Yahoo Finance         │
                 │ • Google OAuth          │
                 │ • Gmail SMTP            │
                 │ • users.json            │
                 └─────────────────────────┘


## 🔄 Project Workflow

```text
User Opens Application
          ↓
      Sign Up / Login
          ↓
     CAPTCHA Verification
          ↓
 Password / Google OAuth Login
          ↓
       Email OTP
          ↓
    Authentication Success
          ↓
 Enter logo name
          ↓
 Fetch Historical + Live Data
          ↓
      Data Preprocessing
          ↓
     MinMaxScaler Scaling
          ↓
   Create Sequential Windows
          ↓
       LSTM Prediction
          ↓
   Inverse Transform Result
          ↓
 Compare Predicted & Latest Price
          ↓
   BUY / SELL / HOLD Signal
          ↓
 Charts + Prediction Dashboard

## Data Preprocessing

The stock data is processed before being provided to the LSTM model.

### Steps

1. Retrieve stock data using `yfinance`.
2. Extract the closing-price series.
3. Handle missing values.
4. Split the dataset into training and testing portions.
5. Apply `MinMaxScaler`.
6. Create sequential time-series windows.
7. Pass the prepared data to the LSTM model.
8. Inverse-transform predictions back to the original price scale.

The implementation uses an **80:20 training/testing split** and rolling windows for LSTM input preparation.

## Technologies Used

| Technology            | Purpose                    |
| --------------------- | -------------------------- |
| 🐍 Python             | Main programming language  |
| 🧠 TensorFlow / Keras | LSTM deep learning model   |
| 📊 Pandas             | Data processing            |
| 🔢 NumPy              | Numerical operations       |
| 📉 Scikit-learn       | MinMaxScaler preprocessing |
| 📈 Matplotlib         | Data visualization         |
| 💹 yfinance           | Yahoo Finance stock data   |
| 🌐 Streamlit          | Web application interface  |
| 🔐 CAPTCHA            | Bot protection             |
| 📧 Gmail SMTP         | OTP email delivery         |
| 🔑 Google OAuth 2.0   | Google authentication      |
| 💾 JSON               | Local user data storage    |
| 💻 VS Code            | Development environment    |

These technologies are documented in the project's technical requirements and implementation report.

## Application Dashboard

The application provides:

* Live stock price information
* Historical price data
* Moving-average charts
* Actual vs predicted price visualization
* Next-day predicted closing price
* BUY / SELL / HOLD recommendation
* Secure authentication workflow

The project report documents the Sign-Up page, Login page, OTP page, prediction dashboard, moving-average charts, and actual-vs-predicted visualization.

---

## Requirements

### Hardware

* Intel Core i3 8th Generation or equivalent AMD Ryzen 3 or higher
* Minimum 4 GB RAM
* 8 GB RAM recommended
* Minimum 5 GB free storage
* Stable internet connection

### Software

* Windows 10/11, macOS 12+, or modern Linux
* Python 3.10+
* Google Chrome / Firefox / Edge / Safari
* Visual Studio Code or another Python IDE

The project report specifies Python 3.10+ and the required Python libraries and external services.

### Installation

### 1. Clone the Repository

```bash
git clone https://github.com/poojitha07124/poojitha07124.git
```

### 2. Navigate to the Project Folder

```bash
cd Stock-Price-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```
The application will open in your browser.
---

## 🔑 Environment Variables

For secure configuration, the project uses environment variables for sensitive credentials.

Example:
```text
APP_GMAIL_USER=your_email@gmail.com
APP_GMAIL_APP_PASSWORD=your_app_password

GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=your_redirect_uri
```
> Never upload passwords, API credentials, OAuth secrets, or Gmail App Passwords to GitHub.

## 📈 Results

The developed system successfully integrates:

* Historical stock-data retrieval
* Data preprocessing
* LSTM-based prediction
* Prediction evaluation
* Stock-price visualization
* Moving-average analysis
* BUY/SELL/HOLD recommendation
* Secure authentication
* Browser-based interaction

The project report states that predicted prices closely followed the real market trend during evaluation and uses RMSE as the primary prediction-error metric.

## 🔮 Future Enhancements

The project can be extended with:

* 📅 Multi-step stock forecasting
* 📰 Financial news sentiment analysis
* 📱 Mobile application deployment
* 🔔 Email/SMS/mobile price alerts
* 🧠 Explainable AI
These enhancements are identified in the project report as possible future extensions.

## 👩‍💻 Author

### Poojitha S

📊 **Data Analyst | Python | SQL | Machine Learning | Power BI**

<p align="center">
  <a href="mailto:poojitha07124@gmail.com">
    <img src="https://img.shields.io/badge/GMAIL-FF3B30?style=for-the-badge&logo=gmail&logoColor=white" />
  </a>
  <a href="https://www.linkedin.com/in/poojitha-s-008091274">
    <img src="https://img.shields.io/badge/LINKEDIN-00AEEF?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="https://github.com/poojitha07124">
    <img src="https://img.shields.io/badge/GITHUB-111111?style=for-the-badge&logo=github&logoColor=00FF66" />
  </a>
</p>
