"""
## Documentation
Quickstart: https://github.com/google/generative-ai-python

## Setup
To install the dependencies for this script, run:
pip install -r requirements.txt
"""

import asyncio
import base64
import io
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

import os
import time
import wave
import speech_recognition as sr
from dotenv import load_dotenv
import numpy as np
import asyncio
import threading
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

MODEL = "gemini-2.0-pro-vision-latest"

DEFAULT_MODE = "camera"

# Initialize the model
model = genai.GenerativeModel('gemini-pro-vision')

# Create a chat session
chat = model.start_chat(history=[])

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            
            # Send text to chat and get response
            response = await asyncio.to_thread(
                chat.send_message,
                text or "."
            )
            print(f"\nGemini: {response.text}\n")

    def _get_frame(self, cap):
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        return img

    async def get_frames(self):
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        screenshot = sct.grab(monitor)
        
        # Convert to PIL Image
        img = PIL.Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        img.thumbnail([1024, 1024])
        return img

    async def get_screen(self):
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            img = await self.out_queue.get()
            # Process the frame with Gemini Vision
            response = await asyncio.to_thread(
                model.generate_content,
                [
                    "Describe what you see in this image.",
                    img
                ]
            )
            print(f"\nGemini Vision: {response.text}\n")

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while True:
            data = await self.audio_in_queue.get()
            print(f"Received audio data: {len(data)} bytes")

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            self.audio_in_queue = asyncio.Queue()
            self.out_queue = asyncio.Queue(maxsize=5)

            async with asyncio.TaskGroup() as tg:
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)

if __name__ == "__main__":
    """
    python test_audio_screen.py --mode screen
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    print(args.mode)
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())