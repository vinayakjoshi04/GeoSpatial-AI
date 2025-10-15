import os
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("Testing Gemini API access...")
print(f"API Key (first 10 chars): {GEMINI_API_KEY[:10]}...")
print("\n" + "="*50)

# Test different API versions to list models
api_versions = [
    "https://generativelanguage.googleapis.com/v1/models",
    "https://generativelanguage.googleapis.com/v1beta/models"
]

for api_url in api_versions:
    print(f"\n📋 Checking models at: {api_url}")
    try:
        response = requests.get(f"{api_url}?key={GEMINI_API_KEY}", timeout=5)
        response.raise_for_status()
        models = response.json()
        
        if 'models' in models:
            # Filter only models that support generateContent
            content_models = [m for m in models['models'] 
                            if 'generateContent' in m.get('supportedGenerationMethods', [])]
            print(f"✅ Found {len(content_models)} models that support generateContent:")
            for model in content_models:
                name = model.get('name', 'Unknown')
                print(f"  ✓ {name}")
        else:
            print("No models found in response")
            
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.response.status_code}")
        print(f"   Response: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*50)
print("\n🧪 Testing actual generation requests...")

# Test actual generation with available models
test_models = [
    ("gemini-2.5-flash", "v1"),
    ("gemini-2.5-pro", "v1"),
    ("gemini-2.0-flash", "v1"),
    ("gemini-2.5-flash", "v1beta"),
    ("gemini-2.0-flash-exp", "v1beta"),
]

for model_name, api_version in test_models:
    model_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
    print(f"\n Testing: {model_url}")
    try:
        data = {
            "contents": [{
                "parts": [{"text": "Say 'Hello' in one word"}]
            }],
            "generationConfig": {
                "maxOutputTokens": 10
            }
        }
        response = requests.post(
            f"{model_url}?key={GEMINI_API_KEY}",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"  ✅ SUCCESS! Response: {text.strip()}")
            print(f"  👉 This model works! Use: {model_name} with API version {api_version}")
            break
        else:
            print(f"  ⚠️ Unexpected response format")
            
    except requests.exceptions.HTTPError as e:
        error_detail = "Unknown error"
        try:
            error_json = e.response.json()
            error_detail = error_json.get('error', {}).get('message', 'Unknown error')
        except:
            error_detail = e.response.text[:100]
        print(f"  ❌ HTTP {e.response.status_code}: {error_detail}")
    except requests.exceptions.Timeout:
        print(f"  ❌ Request timed out")
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")

print("\n" + "="*50)
print("\n📊 Testing your actual gemini_advice.py configuration...")

# Test the exact configuration used in gemini_advice.py (now using gemini-2.0-flash)
test_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
test_prompt = "The temperature is 35°C and AQI is 150. Should I go for a run?"

print(f"\nTesting with real weather question...")
try:
    data = {
        "contents": [{
            "parts": [{"text": test_prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 150,
            "topP": 0.8,
            "topK": 10
        }
    }
    response = requests.post(test_url, json=data, headers={"Content-Type": "application/json"}, timeout=10)
    response.raise_for_status()
    result = response.json()
    
    if 'candidates' in result and len(result['candidates']) > 0:
        advice = result['candidates'][0]['content']['parts'][0]['text']
        print(f"✅ SUCCESS! Your gemini_advice.py configuration works!")
        print(f"\nSample advice received:")
        print("-" * 50)
        print(advice)
        print("-" * 50)
    else:
        print(f"⚠️ Unexpected response: {result}")
        
except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP Error: {e.response.status_code}")
    try:
        error_json = e.response.json()
        print(f"   Error: {error_json.get('error', {}).get('message', 'Unknown')}")
    except:
        print(f"   Response: {e.response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50)
print("\n💡 Summary:")
print("1. If you see '✅ SUCCESS!' above, your gemini_advice.py is ready to use!")
print("2. If you see errors, check:")
print("   - Your API key is valid at https://aistudio.google.com/app/apikey")
print("   - The Gemini API is enabled for your project")
print("   - You have not exceeded rate limits")
print("\n3. Run your Streamlit app with: streamlit run streamlit_app.py")