"""
Check if Gemini API is available and test rate limits
"""
import google.generativeai as genai
import time
from config import APIConfig

print("Checking Gemini API Status...")
print("="*60)

# Configure API
genai.configure(api_key=APIConfig.GEMINI_API_KEY)

# Test different models
models_to_test = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash"
]

for model_name in models_to_test:
    print(f"\nTesting: {model_name}")
    print("-"*60)
    
    try:
        model = genai.GenerativeModel(model_name)
        
        # Try 3 quick requests
        for i in range(3):
            start_time = time.time()
            response = model.generate_content(f"Say 'Test {i+1} successful'")
            elapsed = time.time() - start_time
            
            print(f"  Request {i+1}: ✓ Success ({elapsed:.2f}s)")
            print(f"    Response: {response.text[:50]}")
            
            # Small delay between requests
            time.sleep(2)
        
        print(f"  Status: {model_name} is AVAILABLE")
        
    except Exception as e:
        print(f"  Status: {model_name} is LIMITED")
        print(f"  Error: {str(e)[:100]}")
        
        # Check if it's rate limit error
        if "429" in str(e) or "quota" in str(e).lower():
            print(f"  Reason: Rate limit - wait and try again")
        else:
            print(f"  Reason: Other error")

print("\n" + "="*60)
print("\nRecommendation:")
print("If all tests passed: Continue with development")
print("If rate limited: Wait 5-10 minutes and run this script again")