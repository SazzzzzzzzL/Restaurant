import streamlit as st
import pandas as pd
import numpy as np
import pickle
from haversine import haversine, Unit
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# --- Function to load data and model (cached for performance) ---
@st.cache_resource
def load_data_and_model():
    """Loads all necessary data files and the trained model."""
    try:
        vendors = pd.read_csv('vendors.csv')
        train_customers = pd.read_csv('train_customers.csv')
        train_locations = pd.read_csv('train_locations.csv')
        
        vendors['id'] = pd.to_numeric(vendors['id'], errors='coerce')
        vendors['id'] = vendors['id'].fillna(-1).astype(str)
        
        with open('restaurant_recommender_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        le = LabelEncoder()
        vendors['vendor_category_en'] = vendors['vendor_category_en'].fillna('Unknown')
        vendors['vendor_category_encoded'] = le.fit_transform(vendors['vendor_category_en'])
        
        train_customers['dob'] = pd.to_datetime(train_customers['dob'], errors='coerce')
        train_customers['age'] = (pd.to_datetime('now', utc=True) - train_customers['dob'].dt.tz_localize('UTC')).dt.days / 365.25
        train_customers['age'] = train_customers['age'].fillna(train_customers['age'].median())
        
        return vendors, train_customers, train_locations, model
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure all data files and the model file are in the same directory.")
        return None, None, None, e

# Load data and model once
vendors, customers, locations, model = load_data_and_model()
if vendors is None:
    st.stop()

# --- App Title and Description ---
st.title("🍔 Restaurant Recommendation A/B Testing Simulator")
st.markdown("""
This tool allows business owners to simulate hypothetical changes to their restaurant's profile and see the predicted impact on their recommendation score.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Select Your Restaurant")
selected_vendor_id = st.sidebar.selectbox(
    "Choose a Vendor ID", 
    vendors['id'].unique()
)

# --- Core Function for What-If Analysis ---
def run_what_if_analysis(selected_vendor_id):
    random_customer_loc = locations.sample(1).iloc[0]
    # --- FIX: Use 'customer_id' instead of 'CID' ---
    random_customer = customers[customers['customer_id'] == random_customer_loc['customer_id']].iloc[0]
    
    selected_vendor = vendors[vendors['id'] == str(selected_vendor_id)].iloc[0]

    cust_loc = (random_customer_loc['latitude'], random_customer_loc['longitude'])
    vendor_loc = (selected_vendor['latitude'], selected_vendor['longitude'])
    distance_km = haversine(cust_loc, vendor_loc, unit=Unit.KILOMETERS)

    base_features = [
        distance_km,
        selected_vendor['vendor_category_encoded'],
        random_customer['age']
    ]

    base_score = model.predict_proba([base_features])[:, 1][0]
    hypo_score = base_score

    fig, ax = plt.subplots(facecolor='#FFFFFF')
    scores = [base_score, hypo_score]
    labels = ['Current', 'Hypothetical']
    ax.bar(labels, scores, color=['#7FD2C8', '#FF9F9F'])
    ax.set_ylim(0, 1)
    ax.set_title("Recommendation Score Comparison")
    ax.set_ylabel("Predicted Score")
    st.pyplot(fig)
    
    st.markdown("### Business Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Predicted Score", f"{base_score:.4f}")
    with col2:
        st.metric("Hypothetical Predicted Score", f"{hypo_score:.4f}")

# --- Core Function for New Location Analysis ---
def run_new_location_analysis(selected_vendor_id, new_cust_lat, new_cust_lon):
    # --- FIX: Use 'customer_id' instead of 'CID' ---
    random_customer = customers.sample(1).iloc[0]
    selected_vendor = vendors[vendors['id'] == str(selected_vendor_id)].iloc[0]

    cust_loc = (new_cust_lat, new_cust_lon)
    vendor_loc = (selected_vendor['latitude'], selected_vendor['longitude'])
    new_distance_km = haversine(cust_loc, vendor_loc, unit=Unit.KILOMETERS)
    
    new_loc_features = [
        new_distance_km,
        selected_vendor['vendor_category_encoded'],
        random_customer['age']
    ]
    
    new_loc_score = model.predict_proba([new_loc_features])[:, 1][0]

    st.markdown(f"""
    ### New Customer Location Analysis
    - **Distance to Restaurant:** {new_distance_km:.2f} km
    - **Predicted Score for New Location:** {new_loc_score:.4f}
    """)

# --- UI for What-If Analysis ---
st.header("1. What-If Analysis: Adjust Your Profile")
st.markdown("Use the sliders to simulate changes to your restaurant's details.")

col1, col2 = st.columns(2)
with col1:
    rating_slider = st.slider("Vendor Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1, help="Simulate a change in your average rating.", disabled=True)
with col2:
    delivery_charge_slider = st.slider("Delivery Charge ($)", min_value=0, max_value=20, value=5, step=1, help="See how a price change affects visibility.", disabled=True)

if st.button("Run What-If Analysis"):
    run_what_if_analysis(selected_vendor_id)

st.markdown("---")

# --- UI for New Location Analysis ---
st.header("2. New Customer Location Analysis")
st.markdown("Simulate a new customer in a specific location to see your restaurant's potential reach.")

col1, col2 = st.columns(2)
with col1:
    new_cust_lat = st.number_input("Customer Latitude", value=30.0, step=0.1)
with col2:
    new_cust_lon = st.number_input("Customer Longitude", value=31.0, step=0.1)

if st.button("Analyze New Location"):
    run_new_location_analysis(selected_vendor_id, new_cust_lat, new_cust_lon)
