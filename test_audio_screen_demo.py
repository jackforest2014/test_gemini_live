import os
import time
import wave
import pyaudio
import speech_recognition as sr
from dotenv import load_dotenv
import cv2
import numpy as np
import mss
import PIL.Image
import io
import base64
import asyncio
import threading
import argparse
import pytesseract
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "temp_recording.wav"

# Screenshot parameters
SCREENSHOT_INTERVAL = 1.0  # Take a screenshot every 1 second

def record_audio():
    """Record audio from microphone and save to WAV file"""
    print("\nRecording... Speak now!")
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        try:
            data = stream.read(CHUNK)
            frames.append(data)
        except Exception as e:
            print(f"Error during recording: {e}")
            continue
    
    print("Done recording!")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return WAVE_OUTPUT_FILENAME

def transcribe_audio(audio_file):
    """Transcribe audio file to text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"\nTranscribed text: {text}")
            return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def capture_screenshot():
    """Capture a screenshot of the entire screen"""
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[0]
        screenshot = sct.grab(monitor)
        
        # Convert to PIL Image
        img = PIL.Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        
        # Save the screenshot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        img.save(filename)
        
        print(f"Screenshot saved as {filename}")
        return filename

def capture_screenshots_thread(stop_event):
    """Thread function to capture screenshots at regular intervals"""
    while not stop_event.is_set():
        capture_screenshot()
        time.sleep(SCREENSHOT_INTERVAL)

def extract_text_from_image(image_path):
    """Extract text from an image using OCR"""
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Perform OCR on the image
        text = pytesseract.image_to_string(img)
        
        # Split the text into lines
        lines = text.split('\n')
        
        # Filter out empty lines
        lines = [line for line in lines if line.strip()]
        
        # Try to identify problem statement and code
        problem_statement = []
        code_snippet = []
        in_code_section = False
        
        for line in lines:
            # Check if this line might be code (contains common programming keywords or symbols)
            is_code = any(keyword in line for keyword in ['def', 'class', 'import', 'return', 'if', 'else', 'for', 'while', '(', ')', '{', '}', ';', '='])
            
            if is_code:
                in_code_section = True
                code_snippet.append(line)
            elif in_code_section:
                # If we were in a code section but this line doesn't look like code,
                # we might be back to the problem statement
                if len(line) > 20:  # Problem statements tend to be longer
                    in_code_section = False
                    problem_statement.append(line)
                else:
                    code_snippet.append(line)
            else:
                problem_statement.append(line)
        
        # If we couldn't identify a clear separation, assume the first part is the problem
        # and the rest is code
        if not problem_statement and not code_snippet:
            mid_point = len(lines) // 2
            problem_statement = lines[:mid_point]
            code_snippet = lines[mid_point:]
        
        return {
            'full_text': text,
            'problem_statement': '\n'.join(problem_statement),
            'code_snippet': '\n'.join(code_snippet)
        }
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None

def analyze_image(image_path):
    """Analyze an image using Google's Gemini Vision model"""
    try:
        total_start_time = time.time()
        
        # Load the image
        load_start = time.time()
        img = PIL.Image.open(image_path)
        load_time = time.time() - load_start
        
        # Initialize Gemini model
        print(f"Using Gemini API key: {GEMINI_API_KEY[:5]}...{GEMINI_API_KEY[-5:] if len(GEMINI_API_KEY) > 10 else ''}")
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        # Prepare the image for Gemini
        prep_start = time.time()
        # Convert PIL Image to bytes
        image_bytes = io.BytesIO()
        img.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()
        
        # Create image parts for the model
        image_parts = [
            {
                "mime_type": "image/png",
                "data": base64.b64encode(image_bytes).decode()
            }
        ]
        prep_time = time.time() - prep_start
        
        # Generate content with Gemini
        prompt = "Extract the problem statement and code snippet from the image, and the selected line (the background color is different from the rest of the code snippet) of code from the code snippet."
        
        # Add retry logic for API calls
        max_retries = 3
        retry_delay = 2  # seconds
        description = "API connection failed. Please check your internet connection and API key."
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to Gemini API (attempt {attempt+1}/{max_retries})...")
                api_start = time.time()
                response = model.generate_content([prompt, image_parts[0]])
                api_time = time.time() - api_start
                description = response.text
                
                # Calculate total time
                total_time = time.time() - total_start_time
                
                # Print timing breakdown
                print("\nTiming Analysis:")
                print(f"├── Image Loading: {load_time:.3f} seconds")
                print(f"├── Image Preparation: {prep_time:.3f} seconds")
                print(f"├── API Call: {api_time:.3f} seconds")
                print(f"└── Total Processing: {total_time:.3f} seconds")
                
                print(f"\nGemini Image Analysis:")
                print(description)
                break
            except Exception as e:
                error_str = str(e)
                print(f"Error details: {error_str}")
                
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    if attempt < max_retries - 1:
                        print(f"API connection error (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("Failed to connect to Gemini API after multiple attempts. Check your internet connection and API key.")
                elif "API key" in error_str or "authentication" in error_str:
                    print("Authentication error. Please check your API key.")
                    description = "Authentication error. Please check your API key."
                    break
                else:
                    print(f"Error with Gemini API: {e}")
                    description = f"Error: {str(e)}"
                break
        
        # Extract text from the image using OCR as backup
        print("\nExtracting text from the image...")
        # text_result = extract_text_from_image(image_path)
        
        # if text_result:
        #     return {
        #         'gemini_analysis': description,
        #         'ocr_text': text_result
        #     }
        return {
            'gemini_analysis': description,
            'ocr_text': None
        }
        
    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        # Fallback to OCR only
        # text_result = extract_text_from_image(image_path)
        return {
            'gemini_analysis': None,
            'ocr_text': None
        }

def main():
    """Main function to run the audio recording and screenshot capture"""
    parser = argparse.ArgumentParser(description='Record audio, capture screenshots, or analyze existing images')
    parser.add_argument('--mode', choices=['audio', 'screen', 'both'], default='both',
                      help='Mode to run: audio, screen, or both')
    parser.add_argument('--image', type=str, help='Path to an existing image to analyze')
    args = parser.parse_args()
    
    # If image path is provided, analyze it directly
    if args.image:
        if os.path.exists(args.image):
            print(f"\nAnalyzing image: {args.image}")
            analysis_result = analyze_image(args.image)
            
            if analysis_result['gemini_analysis']:
                print("\n=== Gemini Analysis ===")
                print(analysis_result['gemini_analysis'])
            
            # if analysis_result['ocr_text']:
            #     print("\n=== OCR Results ===")
            #     print("Problem Statement:")
            #     print(analysis_result['ocr_text']['problem_statement'])
            #     print("\nCode Snippet:")
            #     print(analysis_result['ocr_text']['code_snippet'])
                
            #     # Save the extracted text to files
            #     base_name = os.path.splitext(args.image)[0]
            #     with open(f"{base_name}_problem.txt", "w") as f:
            #         f.write(analysis_result['ocr_text']['problem_statement'])
            #     with open(f"{base_name}_code.txt", "w") as f:
            #         f.write(analysis_result['ocr_text']['code_snippet'])
            #     print(f"\nExtracted text saved to {base_name}_problem.txt and {base_name}_code.txt")
        else:
            print(f"Error: Image file '{args.image}' does not exist.")
            return
    
    # Continue with audio/screen capture if no image is provided or if mode is specified
    if not args.image:
        if args.mode in ['audio', 'both']:
            # Record and transcribe audio
            audio_file = record_audio()
            transcribed_text = transcribe_audio(audio_file)
            
            # Clean up the audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)
        
        if args.mode in ['screen', 'both']:
            # Set up screenshot thread
            stop_event = threading.Event()
            screenshot_thread = threading.Thread(target=capture_screenshots_thread, args=(stop_event,))
            
            try:
                # Start screenshot thread
                screenshot_thread.start()
                
                # If we're only doing screenshots, wait for 5 seconds
                if args.mode == 'screen':
                    time.sleep(RECORD_SECONDS)
                
                # Stop the screenshot thread
                stop_event.set()
                screenshot_thread.join()
                
                # Analyze the last screenshot
                screenshots = [f for f in os.listdir('.') if f.startswith('screenshot_')]
                if screenshots:
                    latest_screenshot = max(screenshots, key=os.path.getctime)
                    print(f"\nAnalyzing the latest screenshot: {latest_screenshot}")
                    
                    analysis_result = analyze_image(latest_screenshot)
                    
                    if analysis_result['gemini_analysis']:
                        print("\n=== Gemini Analysis ===")
                        print(analysis_result['gemini_analysis'])
                    
                    # if analysis_result['ocr_text']:
                    #     print("\n=== OCR Results ===")
                    #     print("Problem Statement:")
                    #     print(analysis_result['ocr_text']['problem_statement'])
                    #     print("\nCode Snippet:")
                    #     print(analysis_result['ocr_text']['code_snippet'])
                        
                    #     # Save the extracted text to files
                    #     base_name = os.path.splitext(latest_screenshot)[0]
                    #     with open(f"{base_name}_problem.txt", "w") as f:
                    #         f.write(analysis_result['ocr_text']['problem_statement'])
                    #     with open(f"{base_name}_code.txt", "w") as f:
                    #         f.write(analysis_result['ocr_text']['code_snippet'])
                    #     print(f"\nExtracted text saved to {base_name}_problem.txt and {base_name}_code.txt")
                
            except KeyboardInterrupt:
                print("\nStopping screenshot capture...")
                stop_event.set()
                screenshot_thread.join()

if __name__ == "__main__":
    main() 