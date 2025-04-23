import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_response(prompt):
    """
    Get a response from the Gemini model
    """
    try:
        # For text-only input
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    # Example usage
    prompt = "What is artificial intelligence?"
    response = get_gemini_response(prompt)
    if response:
        print("Gemini Response:")
        print(response)

if __name__ == "__main__":
    main() 