import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Predição - Telco Churn",
    page_icon="📡",
    layout="centered",
)
API_URL = os.getenv("API_URL", "http://localhost:8000")

SIM_NAO = ["No", "Yes"]
SERVICO_INTERNET = ["No", "Yes", "No internet service"]

# Paleta: verde principal, verde escuro, menta de destaque e cinza claro
VERDE = "#3E8E73"
VERDE_ESCURO = "#1F4D3F"
MENTA = "#5FD9AE"

st.markdown(
    f"""
    <style>
    .cabecalho {{
        background: {VERDE};
        border-radius: 24px;
        padding: 2rem 2rem 1.6rem;
        margin-bottom: 1.5rem;
        border-bottom: 6px solid {MENTA};
    }}
    .cabecalho h1 {{
        color: #FFFFFF;
        font-weight: 800;
        margin: 0;
        padding: 0;
    }}
    .cabecalho p {{
        color: #E6F5EF;
        margin: 0.4rem 0 0;
        font-size: 1rem;
    }}
    h2, h3 {{ color: {VERDE}; }}
    .stButton > button, .stFormSubmitButton > button {{
        border-radius: 999px;
        border: 2px solid {VERDE};
        font-weight: 700;
    }}
    .stFormSubmitButton > button {{
        background: {VERDE};
        color: #FFFFFF;
    }}
    .stFormSubmitButton > button:hover {{
        background: {VERDE_ESCURO};
        border-color: {VERDE_ESCURO};
        color: #FFFFFF;
    }}
    </style>
    <div class="cabecalho">
        <h1>Predição - Telco Churn</h1>
        <p>Autores: Greyce, Luiz Felipe, Natalia, Pedro Antonio e Thiago Ney</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Login
if "token" not in st.session_state:
    st.session_state.token = None

if st.session_state.token is None:
    with st.form("login"):
        username = st.text_input("User")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            try:
                r = requests.post(
                    f"{API_URL}/login",
                    json={"username": username, "password": password},
                    timeout=60,
                )
            except requests.ConnectionError:
                st.error(f"Unable to reach API in {API_URL}")
            else:
                if r.status_code == 200:
                    st.session_state.token = r.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Incorrect User or Password")
else:
    st.success("User authenticated")

    with st.form("Client"):
        st.subheader("Client Profile")
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior = col2.selectbox("SeniorCitizen - 1 for yes", [1, 0])
        partner = col1.selectbox("Has a partner?", SIM_NAO)
        dependents = col2.selectbox("Has dependents?", SIM_NAO)

        st.subheader("Services")
        col1, col2 = st.columns(2)
        phone = col1.selectbox("Telephone", SIM_NAO, index=1)
        multiple_lines = col2.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = col1.selectbox("Internet", ["DSL", "Fiber optic", "No"], index=1)
        online_security = col2.selectbox("Online Security", SERVICO_INTERNET)
        online_backup = col1.selectbox("Online Backup", SERVICO_INTERNET)
        device_protection = col2.selectbox("Device Protection", SERVICO_INTERNET)
        tech_support = col1.selectbox("Technical support", SERVICO_INTERNET)
        streaming_tv = col2.selectbox("TV Streaming", SERVICO_INTERNET)
        streaming_movies = col1.selectbox("Movies Streaming", SERVICO_INTERNET)

        st.subheader("Contract & Charges")
        col1, col2 = st.columns(2)
        contract = col1.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = col2.selectbox("Digital Billing", SIM_NAO, index=1)
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        col1, col2, col3 = st.columns(3)
        tenure = col1.number_input("Months as a client", 0, 100, 2)
        monthly = col2.number_input("Mensal Charge", 0.0, 500.0, 70.7)
        total = col3.number_input("Total Billing", 0.0, 20000.0, 151.65)

        enviar = st.form_submit_button("Predict Churn Probability")

    if enviar:
        r = requests.post(
            f"{API_URL}/predict",
            json={
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            },
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            timeout=60,
        )
        if r.status_code == 200:
            dados = r.json()
            prob = dados["probabilidade_churn"]
            if dados["churn"]:
                st.error(f"High churn risk: {prob:.1%}")
            else:
                st.success(f"Low churn risk: {prob:.1%}")
            st.bar_chart(
                {"Churn": [prob], "No churn": [1 - prob]},
                color=[VERDE, MENTA],
            )
        elif r.status_code == 401:
            st.session_state.token = None
            st.warning("Expired Session. Login again.")
            st.rerun()
        else:
            st.error(f"Erro {r.status_code}: {r.text}")

    if st.button("Log Out"):
        st.session_state.token = None
        st.rerun()
