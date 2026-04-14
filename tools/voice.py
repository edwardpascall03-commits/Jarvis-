import whisper
import pyaudio
import wave
import os
import tempfile
import ctypes
import warnings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

whisper_model = None

warnings.filterwarnings("ignore")

import ctypes
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = ctypes.cdll.LoadLibrary('libasound.so.2')
asound.snd_lib_error_set_handler(c_error_handler)

def load_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        print("Whisper ready.")
    return whisper_model

def record_audio(sample_rate=16000):
    response = input("Press Enter to record, or type anything to switch to text mode: ").strip().lower()
    if response:
        return "__TEXT_MODE__"

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024
    )

    print("Recording... press Enter to stop.")
    frames = []
    import threading
    stop_flag = threading.Event()

    def wait_for_enter():
        input()
        stop_flag.set()

    t = threading.Thread(target=wait_for_enter)
    t.daemon = True
    t.start()

    while not stop_flag.is_set():
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    print("Processing...")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return tmp.name

def transcribe(audio_path):
    model = load_whisper()
    result = model.transcribe(audio_path)
    os.unlink(audio_path)
    return result["text"].strip()

def speak(text):
    audio = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2"
    )
    
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    for chunk in audio:
        tmp.write(chunk)
    tmp.close()
    os.system(f"mpg123 -q {tmp.name}")
    os.unlink(tmp.name)

def listen_and_transcribe():
    audio_path = record_audio()
    if audio_path == "__TEXT_MODE__":
        return "__TEXT_MODE__"
    return transcribe(audio_path)