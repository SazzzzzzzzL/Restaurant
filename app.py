import gradio as gr
import pandas as pd
import numpy as np
import pickle
from haversine import haversine, Unit
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# --- Function to load data and model (cached for performance) ---
def load_data_and_model():
    """Loads all necessary data files and the trained model."""
    try:
        vendors = pd.read_csv('vendors.csv')
        train_customers = pd.read_csv('train_customers.csv')
        train_locations = pd.read_csv('train_locations.csv')
        
        # Robustly convert 'id' column to string to avoid ValueError
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
        gr.Warning(f"File not found: {e}. Please ensure all data files and the model file are in the same directory.")
        return None, None, None, e

# Load data and model once
vendors, customers, locations, model = load_data_and_model()
if vendors is None:
    raise FileNotFoundError("Application cannot start. Required files are missing.")

# --- Core Function for What-If Analysis ---
def run_what_if_analysis(selected_vendor_id):
    """
    Simulates a hypothetical change and returns the baseline and new scores,
    along with a plot for visualization.
    """
    random_customer_loc = locations.sample(1).iloc[0]
    random_customer = customers[customers['CID'] == random_customer_loc['CID']].iloc[0]
    
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
    
    return f"{base_score:.4f}", f"{hypo_score:.4f}", fig

# --- Core Function for New Location Analysis ---
def run_new_location_analysis(selected_vendor_id, new_cust_lat, new_cust_lon):
    """
    Simulates a new customer location and returns the predicted score and distance.
    """
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

    return f"Distance to new location: {new_distance_km:.2f} km", f"Predicted Score: {new_loc_score:.4f}"

# --- Gradio UI with a Creative Theme ---
css = """
h1 {
    text-align: center;
    font-family: 'Arial', sans-serif;
    color: #333;
}
.gr-box {
    border-color: #7FD2C8 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.HTML("<h1 style='color: #FF6347;'>🍔 Foodie's Forecast: A/B Test Simulator</h1>")
    gr.Markdown("""
    Welcome, restaurant owner! This tool helps you understand how small changes to your profile can impact your visibility and potential for new orders. Use the sections below to run your own experiments.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            vendor_dropdown = gr.Dropdown(vendors['id'].unique(), label="Select Your Restaurant ID", info="Choose your restaurant from the list.")
        with gr.Column(scale=2):
            gr.Image("https://i.imgur.com/5SgBv1S.png", container=False)

    gr.Markdown("---")
    
    with gr.Tab("What-If Analysis: Adjust Your Profile"):
        gr.Markdown("### Simulate changes to your restaurant's details and see the predicted impact on your recommendation score.")
        
        with gr.Row():
            rating_slider = gr.Slider(0, 5, value=4.0, step=0.1, label="Vendor Rating", info="What if your rating increased or decreased?", interactive=False)
            delivery_charge_slider = gr.Slider(0, 20, value=5, step=1, label="Delivery Charge ($)", info="How does a price change affect your visibility?", interactive=False)
            
        with gr.Row():
            btn_what_if = gr.Button("Run What-If Analysis", variant="primary")

        with gr.Row():
            output_current = gr.Textbox(label="Current Predicted Score")
            output_hypo = gr.Textbox(label="Hypothetical Predicted Score")
        
        gr.Plot(label="Score Comparison", elem_id="plot_what_if")
        
        btn_what_if.click(
            fn=run_what_if_analysis,
            inputs=[vendor_dropdown],
            outputs=[output_current, output_hypo, gr.Plot(label="Score Comparison")]
        )
    
    with gr.Tab("New Customer Location Analysis"):
        gr.Markdown("### What would happen if a new customer moved to a different area?")
        
        with gr.Row():
            lat_input = gr.Number(label="New Customer Latitude", value=30.0)
            lon_input = gr.Number(label="New Customer Longitude", value=31.0)
            
        with gr.Row():
            btn_new_loc = gr.Button("Analyze New Location", variant="secondary")

        with gr.Row():
            output_distance = gr.Textbox(label="Distance to Restaurant")
            output_new_score = gr.Textbox(label="Predicted Score for New Location")
            
        btn_new_loc.click(
            fn=run_new_location_analysis,
            inputs=[vendor_dropdown, lat_input, lon_input],
            outputs=[output_distance, output_new_score]
        )

# Launch the app
demo.launch(share=True)
