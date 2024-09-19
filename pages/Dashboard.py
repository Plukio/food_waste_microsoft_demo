import pandas as pd
import streamlit as st
from datetime import timedelta, datetime
import time
import os
import json
from openai import OpenAI

# Simulated GPT API client (replace with actual API client setup)
client = OpenAI(api_key=st.secrets["OpenAI_key"])

SUGGESTION_FILE = "../data/suggest.json"

def classify_food_with_gpt4(df):
    """
    Use GPT-4 to analyze the food waste data and generate an overview and suggestions.
    """
    try:
        # Create a textual summary from the DataFrame
        total_items = len(df)
        total_amount = df['amount'].sum()
        food_summary = df.groupby('food_name')['amount'].sum().reset_index()
        food_summary_list = food_summary.apply(lambda x: f"{x['food_name']}: {x['amount']:.1f} kg", axis=1).tolist()
        food_summary_text = ", ".join(food_summary_list)
        
        # Construct the prompt for GPT-4
        prompt = (
            f"You are a food waste analyzer. Based on the following data, provide an actionable suggestion for the food waste: "
            f"Total items: {total_items}. Total waste amount: {total_amount:.1f} kg. "
            f"Here is a summary of the waste amounts for different food items: {food_summary_text}. "
        )
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        st.error(f"Error in GPT-4 classification: {e}")
        return None

def save_suggestion_to_file(suggestion):
    """
    Save the suggestion to a JSON file with a timestamp.
    """
    suggestion_data = {
        "timestamp": datetime.now().isoformat(),
        "suggestion": suggestion
    }
    with open(SUGGESTION_FILE, 'w') as f:
        json.dump(suggestion_data, f)

def load_suggestion_from_file():
    """
    Load the suggestion from the JSON file if it exists.
    """
    if os.path.exists(SUGGESTION_FILE):
        with open(SUGGESTION_FILE, 'r') as f:
            suggestion_data = json.load(f)
            return suggestion_data
    return None

def get_daily_suggestion(df):
    """
    Get the daily suggestion either from the file or by generating a new one.
    """
    suggestion_data = load_suggestion_from_file()
    
    # Check if the suggestion data exists and is less than 24 hours old
    if suggestion_data:
        timestamp = datetime.fromisoformat(suggestion_data["timestamp"])
        if (datetime.now() - timestamp) < timedelta(hours=24):
            # Suggestion is less than 24 hours old, use it
            return suggestion_data["suggestion"]
    
    # Otherwise, generate a new suggestion
    suggestion = classify_food_with_gpt4(df)
    if suggestion:
        save_suggestion_to_file(suggestion)
    return suggestion

def stream_data_and_analysis(df):
    """
    Stream data and analysis using Streamlit.
    """
    # Get the daily suggestion
    suggestion = get_daily_suggestion(df)
    
    if suggestion:
        # Display the suggestion using markdown for better formatting
        st.markdown("### Overview and Suggestions")
        st.markdown(suggestion)

# Load the data
df = pd.read_csv('../data/synthetic_food_waste_3months.csv')

# Convert the timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get the latest date in the dataset
current_date = datetime.now()

# Convert amount to kilograms
df['amount'] = df['amount'] / 1000

# Define the date range for the last month and 7 days
last_month_date = current_date - timedelta(days=30)
last_week_date = current_date - timedelta(days=7)

# Filter data for the last month and the last 7 days
last_month_data = df[df['timestamp'] >= last_month_date]
this_week_data = df[df['timestamp'] >= last_week_date]
last_week_data = df[(df['timestamp'] >= last_month_date - timedelta(days=7)) & (df['timestamp'] < last_week_date)]

# Calculate the total waste for each food class in the last 7 days
total_waste_this_week = this_week_data.groupby('food_name')['amount'].sum().reset_index()
total_waste_this_week.columns = ['food_name', 'total_waste_this_week']

# Calculate the total waste for each food class in the previous week
total_waste_last_week = last_week_data.groupby('food_name')['amount'].sum().reset_index()
total_waste_last_week.columns = ['food_name', 'total_waste_last_week']

# Merge dataframes to calculate percentage change
merged_df = pd.merge(total_waste_this_week, total_waste_last_week, on='food_name', how='left').fillna(0)

# Calculate the percentage change
merged_df['percent_change'] = ((merged_df['total_waste_this_week'] - merged_df['total_waste_last_week']) / 
                               merged_df['total_waste_last_week']) * 100

# Prepare daily trend data for each food class
daily_trends = last_month_data.groupby(['food_name', last_month_data['timestamp'].dt.date])['amount'].sum().reset_index() 
daily_trends.columns = ['food_name', 'date', 'amount']

# Create a list of daily trends for each food class
trend_data = {}
for food in merged_df['food_name']:
    trend = daily_trends[daily_trends['food_name'] == food]['amount'].tolist()
    trend_data[food] = trend

# Add the trend data to the merged_df
merged_df['daily_trend'] = merged_df['food_name'].map(trend_data)

# Drop the 'total_waste_last_week' column as it's no longer needed
merged_df = merged_df.drop(columns=['total_waste_last_week'])

# Apply styling to percent_change column
def highlight_change(val):
    color = 'red' if val > 0 else 'green'
    return f'color: {color}'

styled_df = merged_df.style.applymap(highlight_change, subset=['percent_change'])

total_waste_this_week = this_week_data['amount'].sum()
total_waste_last_week = last_week_data['amount'].sum()

# Calculate the percentage change from last week
percent_change = ((total_waste_this_week - total_waste_last_week) / total_waste_last_week) * 100

# Calculate the money saved based on waste reduction (assuming 200 baht per kg)
money_saved = total_waste_this_week * 200 

target = 50000

diff_target = ((money_saved - target) / target) * 100

top_waste_food = this_week_data.groupby('food_name')['amount'].sum().reset_index()
top_waste_food = top_waste_food.sort_values(by='amount', ascending=False).iloc[0]
top_food_name = top_waste_food['food_name']
top_food_waste_amount = top_waste_food['amount']

# Streamlit UI
st.title("Food Waste Monitoring")

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    col1.metric(label="Waste This Week", value=f"{total_waste_this_week:.1f} kg", delta=f"{percent_change:.1f}% from last week",delta_color="inverse")
    col2.metric(label="Money Saved This Week", value=f"{money_saved/1000:.0f}K baht", delta=f"{diff_target:.1f}% from target")
    col3.metric(label="Top Wasted Food", value=f"{top_food_name}: {top_food_waste_amount:.1f} kg", delta=f"{top_food_waste_amount:.1f} kg wasted", delta_color="inverse")

styled_df = styled_df.format({
    "percent_change": "{:.1f}%",
    "total_waste_this_week": "{:.1f}"
})

with st.spinner('Analyzing and preparing your food waste records...'):
    with st.expander("Suggestions"):
        stream_data_and_analysis(this_week_data)

# Display the data using st.data_editor with custom column configurations for daily trend
with st.expander("Show data"):
    st.dataframe(
        styled_df,
        column_config={
            "food_name": "Food Waste",
            "total_waste_this_week": st.column_config.NumberColumn(
                "Waste This Week (kg)",
                help="Total amount of food waste in kilograms for the current week.",
                format="%0.1f kg",
            ),
            "percent_change": st.column_config.NumberColumn(
                "Change from Last Week",
                help="Percentage change in food waste compared to last week.",
                format="%0.1f%%",
            ),
            "daily_trend": st.column_config.LineChartColumn(
                "Daily Trend (Last 14 Days)",
                help="Daily food waste trend for the last month.",
                y_min=0,
            ),
        },
        hide_index=True,
    )


