"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:
`uv sync`
"""

import asyncio
import base64
import io
import traceback
import os

# Set OpenCV environment variable to skip authorization request
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

import google.generativeai as genai
import dotenv
import os
dotenv.load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "gemini-pro"

DEFAULT_MODE = "camera"

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

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
            # Use the current Gemini API
            model = genai.GenerativeModel(MODEL)
            response = model.generate_content(text)
            print(f"Gemini: {response.text}")

    def _get_frame(self, cap):
        try:
            # Read the frame
            ret, frame = cap.read()
            # Check if the frame was read successfully
            if not ret:
                return None
            # Fix: Convert BGR to RGB color space
            # OpenCV captures in BGR but PIL expects RGB format
            # This prevents the blue tint in the video feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
            img.thumbnail([1024, 1024])

            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)

            mime_type = "image/jpeg"
            image_bytes = image_io.read()
            return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    async def get_frames(self):
        try:
            # Set the backend to AVFoundation explicitly
            cap = await asyncio.to_thread(
                cv2.VideoCapture, 0, cv2.CAP_AVFOUNDATION
            )  # 0 represents the default camera
            
            if not cap.isOpened():
                print("Error: Could not open camera. Please check camera permissions.")
                return

            while True:
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    print("Error: Could not read frame from camera.")
                    break

                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)

        except Exception as e:
            print(f"Camera error: {str(e)}")
        finally:
            # Release the VideoCapture object
            if 'cap' in locals():
                cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            
            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            # Check the mime type to determine how to handle the data
            if msg["mime_type"] == "image/jpeg":
                # For images, we can send directly as the data is already in the correct format
                model = genai.GenerativeModel(MODEL)
                response = model.generate_content([msg["data"], "What's in this image?"])
                print(f"Gemini: {response.text}")
            elif msg["mime_type"] == "audio/pcm":
                # For audio data, we'll skip processing for now
                # In a future version, we could add audio transcription here
                continue

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            # This is a placeholder for the audio response
            # In the current API, we don't have direct audio responses
            await asyncio.sleep(1.0)

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
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
    uv run main.py --mode screen
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