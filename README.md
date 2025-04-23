# Gemini API Project

This is a simple Python project that demonstrates how to use the Google Gemini API.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get your API key:
- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create an API key
- Copy the API key

4. Configure the API key:
- Rename `.env.example` to `.env`
- Replace `your_api_key_here` with your actual API key

## Usage

Run the main script:
```bash
python main.py
```

## Audio Testing

To test if PyAudio is working correctly:
```bash
python test_audio.py
```
This will record 2 seconds of audio and save it to `test_output.wav`.

### Troubleshooting PyAudio Installation

If you encounter issues installing PyAudio:

#### On macOS:
1. Install PortAudio using Homebrew:
```bash
brew install portaudio
```
2. Install PyAudio with specific paths:
```bash
pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
```

#### On Windows:
1. Download the appropriate PyAudio wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
2. Install using:
```bash
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```
(Replace the filename with the one you downloaded)

## Features

- Simple interface to interact with Gemini API
- Environment variable management for API key
- Error handling for API requests
- Example implementation of text generation
- Audio recording capabilities with PyAudio

## Requirements

- Python 3.7+
- google-genai
- python-dotenv
- pyaudio (for audio functionality) 