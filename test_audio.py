import pyaudio
import wave
import time
import sys

def test_audio():
    """
    Test if PyAudio is working correctly by recording a short audio clip
    """
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "test_output.wav"

    print("Testing PyAudio...")
    print("This will record 2 seconds of audio and save it to test_output.wav")
    print("Please speak into your microphone when prompted...")
    
    try:
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Get device info
        device_count = audio.get_device_count()
        print(f"Found {device_count} audio devices")
        
        # List available input devices
        print("\nAvailable input devices:")
        for i in range(device_count):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"Device {i}: {device_info['name']}")
        
        # Open audio stream with error handling
        try:
            stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            print("Trying with default input device...")
            stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            input_device_index=None)
        
        print("Recording in 3 seconds...")
        time.sleep(3)
        print("Recording...")
        
        frames = []
        
        # Record audio with error handling
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except OSError as e:
                print(f"Warning: {e}")
                # Continue recording despite the error
                continue
        
        print("Done recording!")
        
        # Stop and close the stream
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
        
        print(f"Audio saved to {WAVE_OUTPUT_FILENAME}")
        print("PyAudio test completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("PyAudio test failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_audio()
    sys.exit(0 if success else 1) 