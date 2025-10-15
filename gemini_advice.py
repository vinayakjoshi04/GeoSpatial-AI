import os
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_aqi_category(aqi):
    """Return AQI category and health implications"""
    if aqi <= 50:
        return "Good", "Air quality is satisfactory"
    elif aqi <= 100:
        return "Moderate", "Acceptable for most people"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "People with respiratory conditions may experience symptoms"
    elif aqi <= 200:
        return "Unhealthy", "Everyone may begin to experience health effects"
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert - everyone may experience serious effects"
    else:
        return "Hazardous", "Health warnings of emergency conditions"

def get_advice(question: str, weather: dict, pollution: dict) -> str:
    try:
        # Use the correct model available in your API (v1 API with gemini-2.0-flash)
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        # Get AQI category for better context
        aqi_val = pollution.get('AQI', 0)
        aqi_category, aqi_health = get_aqi_category(aqi_val)
        
        # Determine weather comfort level
        temp = weather.get('Temperature (°C)', 0)
        if temp > 35:
            temp_desc = "very hot (heat stress risk)"
        elif temp > 30:
            temp_desc = "hot (stay hydrated)"
        elif temp < 10:
            temp_desc = "cold (risk of hypothermia)"
        elif temp < 15:
            temp_desc = "cool"
        else:
            temp_desc = "comfortable"
        
        # Create a sophisticated, context-aware prompt with role-playing
        prompt = f"""You are Dr. Anika Sharma, an environmental health specialist with 15 years of experience advising people on weather and air quality impacts. You combine scientific expertise with practical, empathetic advice.

CONTEXT:
Location: India
Current Situation: A person is asking for advice about their activities given current environmental conditions.

USER'S QUESTION: 
"{question}"

ENVIRONMENTAL CONDITIONS ANALYSIS:

🌡️ WEATHER STATUS:
• Temperature: {weather.get('Temperature (°C)', 'N/A')}°C ({temp_desc})
• Humidity: {weather.get('Humidity (%)', 'N/A')}% {'(high - feels sticky)' if weather.get('Humidity (%)', 0) > 70 else '(moderate)' if weather.get('Humidity (%)', 0) > 40 else '(low - dry air)'}
• Wind Speed: {weather.get('Wind Speed (km/h)', 'N/A')} km/h {'(breezy)' if weather.get('Wind Speed (km/h)', 0) > 20 else '(calm)'}
• Pressure: {weather.get('Pressure (hPa)', 'N/A')} hPa

🌫️ AIR QUALITY STATUS:
• Overall AQI: {pollution.get('AQI', 'N/A')} - {aqi_category}
  └─ Health Impact: {aqi_health}
• Fine Particles (PM2.5): {pollution.get('PM2.5', 'N/A')} μg/m³ {'⚠️ HIGH' if pollution.get('PM2.5', 0) > 55 else '✓ acceptable' if pollution.get('PM2.5', 0) <= 35 else 'moderate'}
• Coarse Particles (PM10): {pollution.get('PM10', 'N/A')} μg/m³
• Nitrogen Dioxide (NO2): {pollution.get('NO2', 'N/A')} μg/m³
• Sulfur Dioxide (SO2): {pollution.get('SO2', 'N/A')} μg/m³
• Carbon Monoxide (CO): {pollution.get('CO', 'N/A')} mg/m³
• Ozone (O3): {pollution.get('O3', 'N/A')} μg/m³

YOUR TASK:
Provide personalized advice that:
1. ✓ Directly answers their question with clear YES/NO/MAYBE recommendation
2. ✓ Explains the PRIMARY concern based on the data (temperature, humidity, AQI, or specific pollutant)
3. ✓ Gives SPECIFIC actionable advice (not generic statements)
4. ✓ Mentions any vulnerable groups that should take extra precautions
5. ✓ Suggests practical alternatives or modifications if conditions aren't ideal
6. ✓ Uses conversational, empathetic tone (like talking to a friend)

Keep it concise (4-6 sentences) but informative. Focus on what matters most for their specific question.

YOUR EXPERT ADVICE:"""
        
        # Prepare request body with optimized parameters
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 300,
                "topP": 0.95,
                "topK": 40,
                "candidateCount": 1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add timeout of 15 seconds for more detailed response
        response = requests.post(url, json=data, headers=headers, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the generated text with better error handling
        try:
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    if len(candidate['content']['parts']) > 0:
                        return candidate['content']['parts'][0]['text']
            return generate_fallback_advice(question, weather, pollution, "⚠️ Unexpected API response format")
        except (KeyError, IndexError, TypeError) as e:
            return generate_fallback_advice(question, weather, pollution, f"⚠️ Error parsing response: {str(e)}")
    
    except requests.exceptions.Timeout:
        return generate_fallback_advice(question, weather, pollution, "⏱️ API timeout - using fallback advice")
    
    except requests.exceptions.HTTPError as e:
        error_msg = f"API Error: {e.response.status_code}"
        if e.response.status_code == 429:
            error_msg = "⚠️ Rate limit exceeded. Please try again in a moment."
        elif e.response.status_code == 400:
            error_msg = "⚠️ Invalid API request. Please check your API key."
        return generate_fallback_advice(question, weather, pollution, error_msg)
    
    except Exception as e:
        return generate_fallback_advice(question, weather, pollution, f"⚠️ Error: {str(e)[:100]}")


def generate_fallback_advice(question: str, weather: dict, pollution: dict, error_msg: str = None) -> str:
    """Generate rule-based advice when API fails"""
    advice = []
    
    if error_msg:
        advice.append(f"{error_msg}\n")
    
    # Get comprehensive data
    temp = weather.get('Temperature (°C)', 0)
    humidity = weather.get('Humidity (%)', 0)
    wind = weather.get('Wind Speed (km/h)', 0)
    aqi = pollution.get('AQI', 0)
    pm25 = pollution.get('PM2.5', 0)
    
    # Determine primary concern
    concerns = []
    
    # Temperature analysis
    if temp > 35:
        concerns.append(("temperature", "🌡️ **Heat Warning**: Temperature is dangerously high. Risk of heat stroke and dehydration. Stay indoors during 11 AM - 4 PM, drink water every 30 minutes, and avoid strenuous activities."))
    elif temp > 30:
        concerns.append(("temperature", "🌡️ **Hot Weather**: Stay well-hydrated (drink 3-4 liters of water), use SPF 30+ sunscreen, and schedule outdoor activities for early morning or evening."))
    elif temp < 10:
        concerns.append(("temperature", "🌡️ **Cold Alert**: Bundle up in layers, cover extremities, and limit exposure to prevent hypothermia. Warm beverages recommended."))
    elif temp < 15:
        concerns.append(("temperature", "🌡️ **Cool Weather**: Light jacket recommended, especially for morning and evening hours."))
    
    # AQI analysis with specific guidance
    if aqi > 200:
        concerns.append(("aqi", "🚨 **Unhealthy Air Quality** (AQI {:.0f}): Avoid all outdoor activities. Keep windows closed, use air purifiers with HEPA filters, wear N95 masks if you must go outside. Particularly dangerous for children, elderly, and those with asthma/COPD.".format(aqi)))
    elif aqi > 150:
        concerns.append(("aqi", "⚠️ **Unhealthy for Sensitive Groups** (AQI {:.0f}): Children, elderly, pregnant women, and people with heart/lung conditions should stay indoors. Others should limit prolonged outdoor activities and consider wearing masks.".format(aqi)))
    elif aqi > 100:
        concerns.append(("aqi", "⚠️ **Moderate Air Quality** (AQI {:.0f}): Sensitive individuals should reduce prolonged outdoor exertion. General population can proceed with normal activities but monitor symptoms.".format(aqi)))
    elif aqi <= 50:
        concerns.append(("aqi", "✅ **Excellent Air Quality** (AQI {:.0f}): Perfect conditions for all outdoor activities!".format(aqi)))
    else:
        concerns.append(("aqi", "✅ **Good Air Quality** (AQI {:.0f}): Safe for outdoor activities.".format(aqi)))
    
    # PM2.5 specific warning
    if pm25 > 75:
        concerns.append(("pm25", "😷 **Critical PM2.5 Levels** ({:.1f} μg/m³): Fine particles can penetrate deep into lungs. N95/N99 masks essential if going outdoors. Use indoor air purifiers.".format(pm25)))
    elif pm25 > 55:
        concerns.append(("pm25", "😷 **High PM2.5** ({:.1f} μg/m³): Avoid outdoor exercise, wear masks outdoors, use air purifiers indoors.".format(pm25)))
    
    # Humidity analysis
    if humidity > 80:
        concerns.append(("humidity", "💧 **High Humidity** ({:.0f}%): Muggy conditions increase heat stress. Stay in air-conditioned spaces, avoid heavy exercise, and stay hydrated.".format(humidity)))
    elif humidity < 30:
        concerns.append(("humidity", "💧 **Low Humidity** ({:.0f}%): Dry air can irritate respiratory system. Drink extra water, use moisturizer, and consider a humidifier indoors.".format(humidity)))
    
    # Wind analysis
    if wind > 50:
        concerns.append(("wind", "💨 **Strong Winds** ({:.0f} km/h): High wind warning. Secure loose objects, avoid parking under trees, and be cautious while driving high-profile vehicles.".format(wind)))
    elif wind > 40:
        concerns.append(("wind", "💨 **Breezy Conditions** ({:.0f} km/h): Moderately strong winds. Secure outdoor items and be cautious with umbrellas.".format(wind)))
    
    # Compile advice prioritizing most severe concerns
    if concerns:
        # Sort by severity (aqi and temperature first)
        priority_order = {"aqi": 0, "pm25": 1, "temperature": 2, "humidity": 3, "wind": 4}
        concerns.sort(key=lambda x: priority_order.get(x[0], 99))
        
        for concern_type, concern_text in concerns[:3]:  # Top 3 concerns
            advice.append(concern_text)
    else:
        advice.append("📊 **Conditions Normal**: Weather and air quality are within comfortable ranges. Standard seasonal precautions apply - stay hydrated and use sun protection during daytime.")
    
    # Add personalized closing based on question context
    question_lower = question.lower()
    if any(word in question_lower for word in ['run', 'jog', 'exercise', 'workout', 'gym']):
        if aqi > 150 or temp > 35:
            advice.append("\n💪 **For Exercise**: Current conditions are NOT suitable for outdoor exercise. Consider indoor alternatives like gym workouts, yoga, or home exercises.")
        elif aqi > 100 or temp > 30:
            advice.append("\n💪 **For Exercise**: Limit intensity and duration. Early morning (6-8 AM) offers better conditions. Stay hydrated throughout.")
        else:
            advice.append("\n💪 **For Exercise**: Conditions are suitable for outdoor exercise. Remember to warm up and stay hydrated!")
    
    return "\n\n".join(advice)