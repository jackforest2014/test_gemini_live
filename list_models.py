import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# List all available models
print("Listing all available Gemini models:")
try:
    for model in genai.list_models():
        print(f"\n- {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  Supported generation methods: {model.supported_generation_methods}")
        if hasattr(model, 'supported_input_modalities'):
            print(f"  Supported input modalities: {model.supported_input_modalities}")
        if hasattr(model, 'supported_output_modalities'):
            print(f"  Supported output modalities: {model.supported_output_modalities}")
except Exception as e:
    print(f"Error listing models: {e}") 