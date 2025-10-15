# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from gemini_advice import get_advice

# Page Configuration
st.set_page_config(
    page_title="India Weather & Pollution Advisor",
    layout="wide",
    page_icon="🌤️",
    initial_sidebar_state="expanded"
)

# Load models and scalers with error handling
@st.cache_resource
def load_models():
    try:
        weather_model = joblib.load("models/weather_best_model.pkl")
        weather_scaler = joblib.load("models/weather_scaler.pkl")
        pollution_model = joblib.load("models/pollution_best_model.pkl")
        pollution_scaler = joblib.load("models/pollution_scaler.pkl")
        return weather_model, weather_scaler, pollution_model, pollution_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

weather_model, weather_scaler, pollution_model, pollution_scaler = load_models()

# ------------------------
# Load city coordinates
# ------------------------
@st.cache_data
def load_city_data():
    try:
        coords = pd.read_csv("data/india_cities_latlon.csv")
        return coords
    except Exception as e:
        st.error(f"Error loading city data: {e}")
        st.stop()

coords = load_city_data()

# ------------------------
# Helper: AQI color coding
# ------------------------
def aqi_color(aqi):
    if aqi <= 50:
        return 'green'
    elif aqi <= 100:
        return 'yellow'
    elif aqi <= 150:
        return 'orange'
    elif aqi <= 200:
        return 'red'
    elif aqi <= 300:
        return 'purple'
    else:
        return 'maroon'

def aqi_category(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

# ------------------------
# Sidebar - Only Navigation
# ------------------------
st.sidebar.title("🌤️ Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📍 City Selection & Map", "🔮 Predictions & Advice", "📊 City Comparison"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("📌 Select a page above to navigate through the application")

# ------------------------
# Main Content
# ------------------------

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.title("🌤️ India Weather & Pollution Advisor")
    st.markdown("### AI-Powered Weather & Air Quality Predictions for Indian Cities")
    
    st.markdown("---")
    
    # Introduction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - 🌡️ **Weather Predictions**
          - Temperature
          - Humidity
          - Wind Speed
          - Pressure
        
        - 🌫️ **Air Quality Monitoring**
          - AQI Index
          - PM2.5, PM10
          - NO2, SO2, CO, O3
        """)
    
    with col2:
        st.markdown("### 🤖 AI-Powered Advice")
        st.markdown("""
        - 💡 Personalized health recommendations
        - 🏃 Activity-specific guidance
        - 👨‍👩‍👧‍👦 Family safety tips
        - 😷 Mask recommendations
        - 🏠 Indoor/outdoor activity suggestions
        """)
    
    with col3:
        st.markdown("### 📊 Analysis Tools")
        st.markdown("""
        - 🗺️ Interactive city map
        - 📈 Visual data charts
        - 🏙️ Multi-city comparison
        - 📊 AQI gauge meter
        - 🎨 Color-coded indicators
        """)
    
    st.markdown("---")
    
    # Quick Start Guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        1. **Select a Page** from the sidebar menu
        2. **City Selection & Map** - View all available cities on an interactive map
        3. **Predictions & Advice** - Get weather/pollution forecasts and AI recommendations
        4. **City Comparison** - Compare multiple cities side-by-side
        """)
    
    with col2:
        st.info("""
        **💡 Tip:**
        
        Navigate using the menu on the left sidebar to explore different features!
        """)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏙️ Cities Available", len(coords))
    with col2:
        st.metric("🌍 Coverage", "Pan-India")
    with col3:
        st.metric("🤖 AI Model", "Gemini 2.0")
    with col4:
        st.metric("📈 Data Sources", "ML Models")
    
    st.markdown("---")
    
    # Call to action
    st.success("👈 Use the navigation menu on the left to get started!")

# ==================== CITY SELECTION & MAP PAGE ====================
elif page == "📍 City Selection & Map":
    st.title("📍 City Selection & Interactive Map")
    st.markdown("### Select your city and date for predictions")
    
    st.markdown("---")
    
    # City and Date Selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🌆 Select City")
        city = st.selectbox("Choose a city:", coords['city'].tolist(), key="city_select")
        
        st.markdown("#### 📅 Select Date")
        col_a, col_b = st.columns(2)
        with col_a:
            day = st.number_input("Day", min_value=1, max_value=31, value=15)
            month = st.number_input("Month", min_value=1, max_value=12, value=10)
        with col_b:
            year = st.number_input("Year", min_value=2025, max_value=2030, value=2025)
        
        # Show selected city info
        city_row = coords[coords['city'] == city]
        if not city_row.empty:
            st.success(f"📍 **Selected: {city}**")
            st.info(f"🌐 Latitude: {city_row['latitude'].values[0]:.4f}°\n\n🌐 Longitude: {city_row['longitude'].values[0]:.4f}°")
            
            st.markdown("---")
            st.metric("📅 Selected Date", f"{day}/{month}/{year}")
    
    with col2:
        st.markdown("#### 🗺️ Interactive Map of Indian Cities")
        
        # Create an interactive map with all cities
        fig = go.Figure()
        
        # Add all cities as markers
        fig.add_trace(go.Scattergeo(
            lon=coords['longitude'],
            lat=coords['latitude'],
            text=coords['city'],
            mode='markers',
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(width=1, color='darkblue'),
                opacity=0.7
            ),
            hovertemplate='<b>%{text}</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>',
            name='All Cities'
        ))
        
        # Highlight selected city
        if not city_row.empty:
            fig.add_trace(go.Scattergeo(
                lon=[city_row['longitude'].values[0]],
                lat=[city_row['latitude'].values[0]],
                text=[city],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='star',
                    line=dict(width=3, color='darkred')
                ),
                textposition="top center",
                textfont=dict(size=16, color='red', family='Arial Black'),
                hovertemplate='<b>SELECTED: %{text}</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>',
                name='Selected City'
            ))
        
        # Configure map layout focused on India
        fig.update_geos(
            scope='asia',
            center=dict(lat=20.5937, lon=78.9629),
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showlakes=True,
            lakecolor='rgb(200, 230, 255)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            lataxis_range=[6, 37],
            lonaxis_range=[68, 98]
        )
        
        fig.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Display cities list
    with st.expander("📋 View All Available Cities"):
        cities_df = coords[['city', 'latitude', 'longitude']].sort_values('city')
        st.dataframe(cities_df, use_container_width=True, height=400)
    
    # Store selections in session state
    st.session_state['selected_city'] = city
    st.session_state['selected_day'] = day
    st.session_state['selected_month'] = month
    st.session_state['selected_year'] = year

# ==================== PREDICTIONS & ADVICE PAGE ====================
elif page == "🔮 Predictions & Advice":
    st.title("🔮 Weather & Pollution Predictions")
    
    # Get selections from session state or use defaults
    city = st.session_state.get('selected_city', coords['city'].tolist()[0])
    day = st.session_state.get('selected_day', 15)
    month = st.session_state.get('selected_month', 10)
    year = st.session_state.get('selected_year', 2025)
    
    st.markdown(f"### Predictions for **{city}** on **{day}/{month}/{year}**")
    
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 💬 Ask AI for Personalized Advice")
        question = st.text_area(
            "Type your question here:",
            placeholder="e.g., Should I go for a run today? Is it safe for my kids to play outside? Can I go cycling?",
            height=120
        )
    
    with col2:
        st.markdown("#### ⚙️ Quick Actions")
        st.info(f"📍 **City:** {city}\n\n📅 **Date:** {day}/{month}/{year}")
        
        if st.button("← Change City/Date", use_container_width=True):
            st.info("👈 Use sidebar to navigate to 'City Selection & Map'")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Get Predictions & AI Advice", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing weather and pollution data..."):
            city_row = coords[coords['city'] == city]
            if city_row.empty:
                st.error(f"❌ City {city} not found!")
            else:
                lat = city_row['latitude'].values[0]
                lon = city_row['longitude'].values[0]

                try:
                    # Prepare features for prediction
                    X_input = np.array([[lat, lon, day, month, year]])

                    # Weather prediction
                    X_weather_scaled = weather_scaler.transform(X_input)
                    weather_pred = weather_model.predict(X_weather_scaled)[0]
                    weather_data = {
                        "Temperature (°C)": round(weather_pred[0], 2),
                        "Humidity (%)": round(weather_pred[1], 2),
                        "Wind Speed (km/h)": round(weather_pred[2], 2),
                        "Pressure (hPa)": round(weather_pred[3], 2)
                    }

                    # Pollution prediction
                    X_pollution_scaled = pollution_scaler.transform(X_input)
                    pollution_pred = pollution_model.predict(X_pollution_scaled)[0]
                    pollution_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
                    pollution_data = {col: round(val, 2) for col, val in zip(pollution_cols, pollution_pred)}

                    st.success("✅ Predictions Generated Successfully!")
                    st.markdown("---")

                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display Weather
                        st.markdown("### 🌤️ Weather Forecast")
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("🌡️ Temperature", f"{weather_data['Temperature (°C)']}°C")
                            st.metric("💧 Humidity", f"{weather_data['Humidity (%)']}%")
                        with metric_col2:
                            st.metric("💨 Wind Speed", f"{weather_data['Wind Speed (km/h)']} km/h")
                            st.metric("🔽 Pressure", f"{weather_data['Pressure (hPa)']} hPa")
                    
                    with col2:
                        # Display Pollution with AQI color
                        st.markdown("### 🌫️ Air Quality")
                        aqi_val = pollution_data['AQI']
                        st.markdown(
                            f"**AQI:** <span style='color:{aqi_color(aqi_val)}; font-weight:bold; font-size:32px'>"
                            f"{aqi_val}</span> - {aqi_category(aqi_val)}", 
                            unsafe_allow_html=True
                        )
                        
                        # Display pollution data in a compact format
                        poll_col1, poll_col2, poll_col3 = st.columns(3)
                        with poll_col1:
                            st.metric("PM2.5", f"{pollution_data['PM2.5']}")
                            st.metric("PM10", f"{pollution_data['PM10']}")
                        with poll_col2:
                            st.metric("NO2", f"{pollution_data['NO2']}")
                            st.metric("SO2", f"{pollution_data['SO2']}")
                        with poll_col3:
                            st.metric("CO", f"{pollution_data['CO']}")
                            st.metric("O3", f"{pollution_data['O3']}")
                    
                    st.markdown("---")
                    
                    # Create a pollution gauge chart
                    st.markdown("### 📊 Air Quality Index Meter")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=aqi_val,
                        title={'text': "Air Quality Index (AQI)", 'font': {'size': 26}},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 500], 'tickwidth': 2, 'tickcolor': "darkblue"},
                            'bar': {'color': aqi_color(aqi_val), 'thickness': 0.3},
                            'bgcolor': "white",
                            'borderwidth': 3,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "yellow"},
                                {'range': [100, 150], 'color': "orange"},
                                {'range': [150, 200], 'color': "red"},
                                {'range': [200, 300], 'color': "purple"},
                                {'range': [300, 500], 'color': "maroon"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.8,
                                'value': aqi_val
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20),
                        font={'size': 16}
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown("---")

                    # Gemini advice
                    if question.strip():
                        st.markdown("### 💡 AI-Powered Personalized Advice")
                        with st.spinner("🤖 Consulting AI health advisor..."):
                            advice = get_advice(question, weather_data, pollution_data)
                            st.info(advice)
                    else:
                        st.info("💬 **Tip:** Ask a question above to get personalized health and safety recommendations based on current conditions!")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.exception(e)

# ==================== CITY COMPARISON PAGE ====================
elif page == "📊 City Comparison":
    st.title("📊 Multi-City Comparison")
    
    # Get date from session state or use defaults
    day = st.session_state.get('selected_day', 15)
    month = st.session_state.get('selected_month', 10)
    year = st.session_state.get('selected_year', 2025)
    
    st.markdown(f"### Compare Weather & Air Quality for **{day}/{month}/{year}**")
    
    st.markdown("---")
    
    # Multi-select for cities
    st.markdown("#### 🏙️ Select Cities to Compare")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_cities = st.multiselect(
            "Choose 2-6 cities:",
            coords['city'].tolist(),
            default=[coords['city'].tolist()[0], coords['city'].tolist()[min(1, len(coords)-1)]] if len(coords) > 1 else [coords['city'].tolist()[0]],
            max_selections=6
        )
    
    with col2:
        st.info(f"📅 **Comparison Date:**\n\n{day}/{month}/{year}")
        
        if st.button("← Change Date", use_container_width=True):
            st.info("👈 Navigate to 'City Selection & Map' to change the date")
    
    st.markdown("---")
    
    if len(selected_cities) > 0 and st.button("🔍 Compare Selected Cities", type="primary", use_container_width=True):
        comparison_data = []
        
        with st.spinner(f"🔄 Generating predictions for {len(selected_cities)} cities..."):
            for comp_city in selected_cities:
                city_row = coords[coords['city'] == comp_city]
                if not city_row.empty:
                    lat = city_row['latitude'].values[0]
                    lon = city_row['longitude'].values[0]
                    
                    X_input = np.array([[lat, lon, day, month, year]])
                    
                    # Weather prediction
                    X_weather_scaled = weather_scaler.transform(X_input)
                    weather_pred = weather_model.predict(X_weather_scaled)[0]
                    
                    # Pollution prediction
                    X_pollution_scaled = pollution_scaler.transform(X_input)
                    pollution_pred = pollution_model.predict(X_pollution_scaled)[0]
                    
                    comparison_data.append({
                        'City': comp_city,
                        'Temperature (°C)': round(weather_pred[0], 2),
                        'Humidity (%)': round(weather_pred[1], 2),
                        'Wind Speed (km/h)': round(weather_pred[2], 2),
                        'AQI': round(pollution_pred[6], 2),
                        'PM2.5': round(pollution_pred[0], 2),
                        'PM10': round(pollution_pred[1], 2)
                    })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            st.success(f"✅ Comparison completed for {len(selected_cities)} cities!")
            st.markdown("---")
            
            # Display comparison table
            st.markdown("### 📋 Comparison Table")
            st.dataframe(
                df_comparison.style.background_gradient(cmap='RdYlGn_r', subset=['AQI']).format(precision=2),
                use_container_width=True,
                height=min(400, 50 + len(df_comparison) * 35)
            )
            
            st.markdown("---")
            
            # Create comparison charts
            st.markdown("### 📈 Visual Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Temperature comparison
                fig_temp = px.bar(
                    df_comparison, 
                    x='City', 
                    y='Temperature (°C)', 
                    title='🌡️ Temperature Comparison',
                    color='Temperature (°C)',
                    color_continuous_scale='RdYlBu_r',
                    text='Temperature (°C)'
                )
                fig_temp.update_traces(texttemplate='%{text:.1f}°C', textposition='outside')
                fig_temp.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                # AQI comparison
                fig_aqi = px.bar(
                    df_comparison, 
                    x='City', 
                    y='AQI',
                    title='🌫️ Air Quality Index Comparison',
                    color='AQI',
                    color_continuous_scale=['green', 'yellow', 'orange', 'red', 'purple'],
                    text='AQI'
                )
                fig_aqi.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig_aqi.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_aqi, use_container_width=True)
            
            # PM2.5 and Humidity comparison
            col3, col4 = st.columns(2)
            
            with col3:
                fig_pm25 = px.bar(
                    df_comparison,
                    x='City',
                    y='PM2.5',
                    title='😷 PM2.5 Levels Comparison',
                    color='PM2.5',
                    color_continuous_scale='Reds',
                    text='PM2.5'
                )
                fig_pm25.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig_pm25.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_pm25, use_container_width=True)
            
            with col4:
                fig_humidity = px.bar(
                    df_comparison,
                    x='City',
                    y='Humidity (%)',
                    title='💧 Humidity Comparison',
                    color='Humidity (%)',
                    color_continuous_scale='Blues',
                    text='Humidity (%)'
                )
                fig_humidity.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_humidity.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_humidity, use_container_width=True)
            
            st.markdown("---")
            
            # Find best cities
            st.markdown("### 🏆 Best Cities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                best_aqi_city = df_comparison.loc[df_comparison['AQI'].idxmin(), 'City']
                best_aqi_val = df_comparison.loc[df_comparison['City'] == best_aqi_city, 'AQI'].values[0]
                st.success(f"🌿 **Best Air Quality:** {best_aqi_city} (AQI: {best_aqi_val:.0f})")
            
            with col2:
                best_temp_city = df_comparison.loc[(df_comparison['Temperature (°C)'] - 25).abs().idxmin(), 'City']
                best_temp_val = df_comparison.loc[df_comparison['City'] == best_temp_city, 'Temperature (°C)'].values[0]
                st.success(f"🌡️ **Most Comfortable Temperature:** {best_temp_city} ({best_temp_val:.1f}°C)")
    
    elif len(selected_cities) == 0:
        st.warning("⚠️ Please select at least one city to compare.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p><strong>📌 India Weather & Pollution Advisor</strong></p>
    <p>Predictions are based on machine learning models trained on historical data. 
    For critical decisions, please consult official weather services.</p>
    <p>Made with ❤️ using Streamlit & Gemini AI</p>
    </div>
    """,
    unsafe_allow_html=True
)