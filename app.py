import streamlit as st
import sqlite3
import numpy as np
import pickle
import tensorflow as tf
from streamlit import connection, cursor

@st.cache_resource
def load_ml_components():
    model = tf.saved_model.load('career_prediction_model')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le


if 'page' not in st.session_state:
    st.session_state.page = 'login'


def go_to_login():
    st.session_state.page = 'login'


def go_to_signup():
    st.session_state.page = 'signup'


def go_to_dashboard():
    st.session_state.page = 'dashboard'


def login_page():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if check_login(username, password):
            st.success("Logged in successfully!")
            st.session_state.page = 'dashboard'
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.button("Sign up here", type="tertiary", on_click=go_to_signup)


def signup_page():
    st.title("Signup Page")

    username = st.text_input("New Username")
    password = st.text_input("New Password", type='password')
    email = st.text_input("Email")
    contact_number = st.text_input("Contact Number")

    if st.button("Signup"):
        if username and password:
            if add_user(username, password, email, contact_number):
                st.success("Account created successfully!")
            else:
                st.error("Username already taken.")
        else:
            st.warning("Please fill in the required fields.")

    st.button("Already have an account?",
              type="tertiary", on_click=go_to_login)


def check_login(username, password):
    with sqlite3.connect('my_database.db') as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM user WHERE user_name=? AND password=?', (username, password))
        result = cursor.fetchone()
        return result is not None


def init_db():
    with sqlite3.connect('my_database.db') as connection:
        cursor = connection.cursor()
        create_table_query = '''
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        password TEXT NOT NULL,
        email TEXT,
        contact_number TEXT
    );
    '''

    cursor.execute(create_table_query)
    connection.commit()


def add_user(name, pw, mail, contact):
    """Call this inside the Signup page logic."""
    try:
        with sqlite3.connect('my_database.db') as conn:
            cursor = conn.cursor()
            query = 'INSERT INTO user (user_name, password, email, contact_number) VALUES (?, ?, ?, ?)'
            cursor.execute(query, (name, pw, mail, contact))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False


def dashboard():
    st.title("Career Prediction Dashboard")
    st.write("Welcome to the dashboard! Please fill in your skills and traits to get a career prediction.")

    col1, col2 = st.columns([8, 2])
    with col2:
        st.button("Logout", on_click=go_to_login)

    try:
        model, scaler, le = load_ml_components()
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return

    st.subheader("Technical Skills (Rate from 0 to 6)")

    col_a, col_b = st.columns(2)
    with col_a:
        db_fun = st.slider("Database Fundamentals", 0, 6, 3)
        comp_arch = st.slider("Computer Architecture", 0, 6, 3)
        dist_comp = st.slider("Distributed Computing Systems", 0, 6, 3)
        cyber_sec = st.slider("Cyber Security", 0, 6, 3)
        networking = st.slider("Networking", 0, 6, 3)
        soft_dev = st.slider("Software Development", 0, 6, 3)
        prog_skills = st.slider("Programming Skills", 0, 6, 3)
        proj_mgmt = st.slider("Project Management", 0, 6, 3)
        comp_forensics = st.slider("Computer Forensics Fundamentals", 0, 6, 3)
    with col_b:
        tech_comm = st.slider("Technical Communication", 0, 6, 3)
        ai_ml = st.slider("AI ML", 0, 6, 3)
        soft_eng = st.slider("Software Engineering", 0, 6, 3)
        bus_analysis = st.slider("Business Analysis", 0, 6, 3)
        comm_skills = st.slider("Communication skills", 0, 6, 3)
        data_science = st.slider("Data Science", 0, 6, 3)
        troubleshooting = st.slider("Troubleshooting skills", 0, 6, 3)
        graphics = st.slider("Graphics Designing", 0, 6, 3)

    st.subheader("Psychological Traits (Rate from 0.0 to 1.0)")
    col_c, col_d = st.columns(2)
    with col_c:
        openness = st.slider("Openness", 0.0, 1.0, 0.5)
        conscientousness = st.slider("Conscientousness", 0.0, 1.0, 0.5)
        extraversion = st.slider("Extraversion", 0.0, 1.0, 0.5)
        agreeableness = st.slider("Agreeableness", 0.0, 1.0, 0.5)
        emotional_range = st.slider("Emotional_Range", 0.0, 1.0, 0.5)
    with col_d:
        conversation = st.slider("Conversation", 0.0, 1.0, 0.5)
        openness_change = st.slider("Openness to Change", 0.0, 1.0, 0.5)
        hedonism = st.slider("Hedonism", 0.0, 1.0, 0.5)
        self_enhancement = st.slider("Self-enhancement", 0.0, 1.0, 0.5)
        self_transcendence = st.slider("Self-transcendence", 0.0, 1.0, 0.5)

    if st.button("Predict Career", type="primary"):
        input_data = np.array([[
            db_fun, comp_arch, dist_comp, cyber_sec, networking, soft_dev,
            prog_skills, proj_mgmt, comp_forensics, tech_comm, ai_ml, soft_eng,
            bus_analysis, comm_skills, data_science, troubleshooting, graphics,
            openness, conscientousness, extraversion, agreeableness, emotional_range,
            conversation, openness_change, hedonism, self_enhancement, self_transcendence
        ]])

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_data = scaler.transform(input_data)
            
            infer = model.signatures["serving_default"]
            prediction_dict = infer(tf.constant(scaled_data, dtype=tf.float32))
            prediction = list(prediction_dict.values())[0].numpy()
        
        predicted_class_idx = np.argmax(prediction, axis=1)
        confidence_score = np.max(prediction, axis=1)[0]

        if hasattr(le, 'categories_'):
            predicted_role = le.categories_[0][predicted_class_idx[0]]
        elif hasattr(le, 'classes_'):
            predicted_role = le.classes_[predicted_class_idx[0]]
        else:
            predicted_role = str(predicted_class_idx[0])

        st.success(f"Predicted Career Path: **{predicted_role}**")
        st.info(f"Confidence Level: **{confidence_score * 100:.2f}%**")


if __name__ == "__main__":
    init_db()
    if st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'signup':
        signup_page()
    elif st.session_state.page == 'dashboard':
        dashboard()
