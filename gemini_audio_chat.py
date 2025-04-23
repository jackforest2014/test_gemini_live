import os
import time
import wave
import pyaudio
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "temp_recording.wav"

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

def get_gemini_response(text):
    """Get response from Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(text)
        return response.text
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return None

def main():
    print("Welcome to Gemini Audio Chat!")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            input("\nPress Enter to start recording...")
            
            # Record audio
            audio_file = record_audio()
            
            # Transcribe audio to text
            transcribed_text = transcribe_audio(audio_file)
            
            if transcribed_text:
                # Get response from Gemini
                print("\nGetting response from Gemini...")
                response = get_gemini_response(transcribed_text)
                
                if response:
                    print("\nGemini's response:")
                    print(response)
            
            # Clean up the temporary audio file
            try:
                os.remove(audio_file)
            except:
                pass
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 