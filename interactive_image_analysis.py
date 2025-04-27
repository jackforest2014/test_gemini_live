"""
Interactive Image Analysis with Gemini API

This script allows you to:
1. Select an image file interactively
2. Enter a custom prompt for analyzing the image
3. Get the response from Gemini API
"""

import os
import io
import base64
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import tkinter as tk
from tkinter import filedialog

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
MODEL = "models/gemini-2.0-flash-live-001"

async def analyze_image_with_prompt(image_path, prompt):
    """Analyze an image with a custom prompt using the Gemini API"""
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' does not exist.")
            return
        
        # Load the image
        img = PIL.Image.open(image_path)
        
        # Resize if needed
        img.thumbnail([1024, 1024])
        
        print(f"\nAnalyzing image: {image_path}")
        print(f"Prompt: {prompt}")
        print("\nWaiting for Gemini response...\n")
        
        # Create a model instance
        model = genai.GenerativeModel(MODEL)
        
        # Generate content with the image and prompt
        response = model.generate_content([prompt, img])
        
        # Print the response
        print(f"Gemini Response:\n{response.text}\n")
    
    except Exception as e:
        print(f"Error analyzing image: {e}")

def select_image_file():
    """Open a file dialog to select an image file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    return file_path

async def interactive_session():
    """Run an interactive session for image analysis"""
    print("=== Interactive Image Analysis with Gemini API ===")
    print("Type 'exit' to quit the program")
    
    while True:
        # Get image file
        print("\nPlease select an image file...")
        image_path = select_image_file()
        
        if not image_path:
            print("No file selected. Exiting...")
            break
        
        # Get prompt
        prompt = input("\nEnter your prompt for analyzing the image (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        
        # Analyze the image
        await analyze_image_with_prompt(image_path, prompt)
        
        # Ask if user wants to continue
        continue_analysis = input("\nDo you want to analyze another image? (y/n): ")
        if continue_analysis.lower() != 'y':
            break
    
    print("\nThank you for using Interactive Image Analysis!")

if __name__ == "__main__":
    asyncio.run(interactive_session()) 