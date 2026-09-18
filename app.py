import os
import json
import ssl
import smtplib
import hashlib
import secrets

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from datetime import datetime, timedelta, date
from email.message import EmailMessage
from captcha.image import ImageCaptcha

from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as requests_lib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense


# ============================================================
# CONFIGURATION
# ============================================================

USERS_FILE = "users.json"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

SMTP_USER = os.environ["APP_GMAIL_USER"]
SMTP_PASS = os.environ["APP_GMAIL_APP_PASSWORD"]

GOOGLE_CID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CS = os.environ["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = os.environ["GOOGLE_REDIRECT_URI"]

OTP_TTL_MIN = 5


# ============================================================
# PASSWORD AUTHENTICATION
# ============================================================

def _hash(value):
    """Generate SHA-256 hash for a given string."""
    return hashlib.sha256(value.encode()).hexdigest()


def load_users():
    """Load registered users from users.json."""
    if not os.path.exists(USERS_FILE):
        return {}

    with open(USERS_FILE, "r") as file:
        return json.load(file)


def save_users(users):
    """Save users to users.json."""
    with open(USERS_FILE, "w") as file:
        json.dump(users, file, indent=2)


def signup(username, password, email):
    """
    Create a new user account.

    Requirements:
    - Username must be at least 3 characters.
    - Password must be at least 4 characters.
    - Email must be a Gmail address.
    """

    users = load_users()

    if username in users:
        return False, "Username already exists."

    if len(username) < 3 or len(password) < 4:
        return False, "Username >= 3 chars, password >= 4 chars."

    if not email.lower().endswith("@gmail.com"):
        return False, "A valid Gmail address is required."

    users[username] = {
        "password_hash": _hash(password),
        "email": email,
        "google_sub": None,
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None
    }

    save_users(users)

    return True, "Account created! Please log in."


def login(username, password):
    """Authenticate a user using username and password."""

    users = load_users()
    user = users.get(username)

    return bool(
        user and
        user.get("password_hash") == _hash(password)
    )


# ============================================================
# CAPTCHA GENERATION AND VERIFICATION
# ============================================================

def new_captcha():
    """
    Generate a new five-character alphanumeric CAPTCHA.
    The generated CAPTCHA is stored in Streamlit session state.
    """

    characters = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    code = "".join(
        secrets.choice(characters)
        for _ in range(5)
    )

    image = ImageCaptcha(
        width=220,
        height=80
    ).generate(code)

    st.session_state["captcha_code"] = code

    return image


def verify_captcha(user_input):
    """Verify CAPTCHA input using a case-insensitive comparison."""

    return (
        (user_input or "").strip().upper()
        == st.session_state.get("captcha_code", "").upper()
    )


# ============================================================
# OTP GENERATION AND EMAIL DISPATCH
# ============================================================

def issue_otp(username, email):
    """
    Generate and send a six-digit OTP.

    The OTP hash and expiry time are stored in Streamlit
    session state instead of storing the actual OTP.
    """

    code = f"{secrets.randbelow(1_000_000):06d}"

    st.session_state["otp"] = {
        "user": username,
        "hash": _hash(code),
        "expires": datetime.utcnow() + timedelta(
            minutes=OTP_TTL_MIN
        ),
        "attempts": 0
    }

    send_otp_email(email, code)


def send_otp_email(to_email, code):
    """Send OTP to the user's registered Gmail address."""

    message = EmailMessage()

    message["Subject"] = "Your Stock Predictor OTP"
    message["From"] = SMTP_USER
    message["To"] = to_email

    message.set_content(
        f"Your OTP is: {code}\n"
        f"Expires in {OTP_TTL_MIN} minutes."
    )

    context = ssl.create_default_context()

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(message)


def verify_otp(code):
    """
    Verify OTP.

    Conditions:
    - OTP must exist.
    - OTP must not be expired.
    - Maximum of 3 attempts are allowed.
    """

    record = st.session_state.get("otp")

    if not record:
        return False, "No OTP issued."

    if datetime.utcnow() > record["expires"]:
        return False, "OTP expired."

    record["attempts"] += 1

    if record["attempts"] > 3:
        return False, "Too many attempts."

    if _hash(code) != record["hash"]:
        return False, "Incorrect OTP."

    return True, "OK"


# ============================================================
# GOOGLE OAUTH 2.0 LOGIN
# ============================================================

def google_flow():
    """Create and configure Google OAuth 2.0 flow."""

    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CID,
                "client_secret": GOOGLE_CS,
                "auth_uri": (
                    "https://accounts.google.com/o/oauth2/auth"
                ),
                "token_uri": (
                    "https://oauth2.googleapis.com/token"
                ),
                "redirect_uris": [REDIRECT_URI]
            }
        },
        scopes=[
            "openid",
            "email",
            "profile"
        ],
        redirect_uri=REDIRECT_URI
    )


def gmail_exchange(auth_code):
    """
    Exchange Google authorization code for user information.

    If the Google account does not already exist,
    a new account is automatically created.
    """

    flow = google_flow()

    flow.fetch_token(code=auth_code)

    info = id_token.verify_oauth2_token(
        flow.credentials.id_token,
        requests_lib.Request(),
        GOOGLE_CID
    )

    email = info["email"]
    google_sub = info["sub"]

    username = email.split("@")[0]

    users = load_users()

    if username not in users:
        users[username] = {
            "password_hash": None,
            "email": email,
            "google_sub": google_sub,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": None
        }

        save_users(users)

    return username, email


# ============================================================
# LSTM MODEL LOADING
# ============================================================

@st.cache_resource
def load_my_model():
    """
    Load the pre-trained LSTM stock prediction model.

    The model is loaded once and cached by Streamlit
    to improve prediction performance.
    """

    model = Sequential([
        Input(shape=(100, 1)),

        LSTM(
            units=128,
            return_sequences=True
        ),

        LSTM(
            units=64,
            return_sequences=False
        ),

        Dense(units=25),

        Dense(units=1)
    ])

    model.load_weights(
        "latest_stock_price_model.keras"
    )

    return model


# ============================================================
# LIVE STOCK DATA FETCHER
# ============================================================

def fetch_live_data(symbol):
    """
    Download historical daily stock data using yfinance.

    When intraday data is available, the latest minute-level
    closing price is used to update today's closing value.
    """

    df = yf.download(
        symbol,
        start="2022-01-01",
        end=date.today().strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    ticker = yf.Ticker(symbol)

    intraday = ticker.history(
        period="1d",
        interval="1m"
    )

    if not intraday.empty:
        last_timestamp = intraday.index[-1]

        last_close = float(
            intraday["Close"].iloc[-1]
        )

        df.loc[
            pd.Timestamp(last_timestamp.date()),
            "Close"
        ] = last_close

    return df


# ============================================================
# DATA PRE-PROCESSING AND INFERENCE
# ============================================================

def prepare_test_data(data, model):
    """
    Prepare stock data for testing and generate predictions.

    The dataset is divided into:
    - 80% training data
    - 20% testing data

    MinMaxScaler is fitted only on the training data
    to reduce the possibility of data leakage.
    """

    close_df = pd.DataFrame(
        data["Close"]
    ).dropna()

    split = int(len(close_df) * 0.80)

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    scaler.fit(
        close_df.iloc[:split]
    )

    past_100 = close_df.iloc[:split].tail(100)

    test_data = pd.concat(
        [
            past_100,
            close_df.iloc[split:]
        ],
        ignore_index=True
    )

    data_test_scale = scaler.transform(
        test_data
    )

    x = []
    y = []

    for i in range(
        100,
        data_test_scale.shape[0]
    ):
        x.append(
            data_test_scale[i - 100:i]
        )

        y.append(
            data_test_scale[i, 0]
        )

    x = np.array(x)
    y = np.array(y)

    predictions = model.predict(
        x,
        verbose=0
    )

    scale_val = 1 / scaler.scale_[0]

    predictions_final = (
        predictions * scale_val
    )

    actual_values = (
        y * scale_val
    )

    return (
        close_df,
        scaler,
        predictions_final,
        actual_values
    )


# ============================================================
# NEXT-STEP PREDICTION AND RECOMMENDATION
# ============================================================

def predict_next_price(close_df, scaler, model):
    """
    Predict the next stock price using the latest
    100-day closing-price window.
    """

    full_scaled = scaler.transform(
        close_df
    )

    last_window = full_scaled[-100:].reshape(
        1,
        100,
        1
    )

    next_prediction = model.predict(
        last_window,
        verbose=0
    )

    scale_val = 1 / scaler.scale_[0]

    next_pred = float(
        next_prediction[0, 0] * scale_val
    )

    last_actual = float(
        close_df["Close"].iloc[-1]
    )

    percentage_change = (
        (next_pred - last_actual)
        / last_actual
        * 100
    )

    if percentage_change > 2:
        recommendation = "BUY"

    elif percentage_change < -2:
        recommendation = "SELL"

    else:
        recommendation = "HOLD"

    return (
        next_pred,
        last_actual,
        percentage_change,
        recommendation
    )
