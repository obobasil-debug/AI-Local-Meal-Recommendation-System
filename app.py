import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Local Meal Recommendation", page_icon="🍲", layout="centered")

# -------------------------
# Sidebar - About
# -------------------------
st.sidebar.title("ℹ️ About App")
st.sidebar.markdown("""
**Local Meal Recommendation System (AI-Based)**

**Developed by:**
- Your Name
- Ogochukwu Offor (NOU211176483)

This app recommends meals based on your health condition and budget.
It also shows foods to avoid for your health condition.
""")

# -------------------------
# Authentication
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin123"}  # default user

def signup():
    st.subheader("📝 Sign Up")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if new_user in st.session_state.users:
            st.error("⚠️ User already exists!")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("✅ User registered! You can now log in.")

def login():
    st.subheader("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.authenticated = True
            st.success(f"🎉 Welcome {username}!")
        else:
            st.error("❌ Invalid credentials!")

# -------------------------
# Auth flow
# -------------------------
if not st.session_state.authenticated:
    option = st.sidebar.selectbox("Select Option", ["Login", "Sign Up"])
    if option == "Login":
        login()
    else:
        signup()

# -------------------------
# Meal Recommendation System
# -------------------------
if st.session_state.authenticated:
    st.title("🍲 Local Meal Recommendation System (AI-Based)")
    st.markdown("---")

    # -------------------------
    # Health conditions
    # -------------------------
    health_conditions = [
        "Diabetes", "Hypertension", "Heart Disease", "Kidney Disease",
        "Ulcer", "IBS", "Lactose Intolerance", "Celiac", "Obesity", "High Cholesterol"
    ]
    selected_health = st.selectbox("Select your health condition:", health_conditions)

    # -------------------------
    # Budget (₦1000–₦10000)
    # -------------------------
    budget = st.slider(
        "Enter your budget amount (₦1000 - ₦10000):",
        min_value=1000, max_value=10000, value=3000, step=100
    )

    # -------------------------
    # Synthetic Dataset
    # -------------------------
    data = pd.DataFrame({
        "meal": [
            "Oatmeal with Fruits","Beans and Plantain","Brown Rice & Vegetable",
            "Eba & Egusi Soup","Vegetable Soup & Fish","Boiled Yam & Sauce",
            "Grilled Fish & Veggies","Boiled Plantain & Beans","Oats & Milk",
            "Moi Moi & Vegetable","Yam Porridge","Okra Soup & Eba",
            "Boiled Rice & Stew","Pap & Moi Moi","Vegetable Porridge",
            "Rice & Grilled Chicken","Oatmeal & Banana","Eba & Vegetable Soup",
            "Plantain Porridge","Vegetable Soup & Fish","Jollof Rice & Chicken",
            "Brown Rice & Beans","Vegetable Salad & Fish","Yam & Egg Sauce",
            "Vegetable Soup & Fish","Boiled Yam & Beans","Oatmeal & Fruit Salad",
            "Grilled Chicken & Salad","Vegetable Soup & Fish","Beans & Plantain"
        ],
        "health": [
            "Diabetes","Diabetes","Diabetes",
            "Hypertension","Hypertension","Hypertension",
            "Heart Disease","Heart Disease","Heart Disease",
            "Kidney Disease","Kidney Disease","Kidney Disease",
            "Ulcer","Ulcer","Ulcer",
            "IBS","IBS","IBS",
            "Lactose Intolerance","Lactose Intolerance","Lactose Intolerance",
            "Celiac","Celiac","Celiac",
            "Obesity","Obesity","Obesity",
            "High Cholesterol","High Cholesterol","High Cholesterol"
        ],
        "cost":[5000,4000,4500,4000,3500,3800,4200,3900,3600,5000,4800,4700,3000,2800,3200,
                3400,3100,3300,3600,3500,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600],
        "avoid":[
            "Sugary foods","Fried foods","Excess oil",
            "High salt","Processed foods","Sugary drinks",
            "Fried foods","High fat","Processed snacks",
            "Too much protein","High salt","Sugary foods",
            "Spicy foods","Fried foods","Acidic foods",
            "High fat","Dairy","Spicy foods",
            "Milk","Cheese","Butter",
            "Wheat","Bread","Pasta",
            "Fried foods","Sugary snacks","High calorie foods",
            "Fried foods","Red meat","High fat foods"
        ]
    })

    # -------------------------
    # Encode health
    # -------------------------
    encoder = LabelEncoder()
    data["health_encoded"] = encoder.fit_transform(data["health"])
    selected_health_encoded = encoder.transform([selected_health])[0]

    # -------------------------
    # Train or load model
    # -------------------------
    MODEL_FILE = "rf_meal_model.pkl"
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        X = data[["health_encoded", "cost"]]
        y = data["meal"]
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)

    # -------------------------
    # Filter meals
    # -------------------------
    budget_range = (budget * 0.8, budget * 1.2)
    filtered_data = data[
        (data["health_encoded"] == selected_health_encoded) &
        (data["cost"] >= budget_range[0]) &
        (data["cost"] <= budget_range[1])
    ]

    # -------------------------
    # AI-style output
    # -------------------------
    if not filtered_data.empty:
        st.subheader("🤖 AI Meal Recommendation:")
        for idx, row in filtered_data.iterrows():
            st.markdown(f"💡 **Meal:** {row['meal']}  |  **Cost:** ₦{row['cost']}")
            st.markdown(f"⚠️ **Foods to avoid:** {row['avoid']}")
            st.markdown("---")
    else:
        st.warning("⚠️ No meals match your health condition and budget. Try adjusting the budget or health condition.")

    # -------------------------
    # Logout
    # -------------------------
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()
