import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")

# Print the API key (first 5 and last 5 characters for security)
if api_key:
    print(f"API key loaded: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
else:
    print("API key not found in .env file") 