import os
import re
import tempfile
import warnings
import logging
import whisper
import fasttext
import sounddevice as sd
import soundfile as sf
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from googletrans import Translator

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("whisper").setLevel(logging.ERROR)

print("🚀 Loading AI models... please wait")
model = whisper.load_model("small")              # lightweight, CPU friendly
lang_model = fasttext.load_model("lid.176.ftz")  # offline language ID
translator = Translator()
print("✅ Models loaded successfully!\n")


def detect_smart_language(text, whisper_lang=None):
    text = text.strip().lower()
    if not text:
        return "en", 0.0

    try:
        ft_label, ft_prob = lang_model.predict(text)
        ft_lang = ft_label[0].replace("__label__", "")
    except Exception:
        ft_lang, ft_prob = "en", 0.0

    if re.search(r'[\u0900-\u097F]+', text):  # Hindi script
        return "hi", 1.0
    if re.search(r'[\u0A00-\u0A7F]+', text):  # Gurmukhi script (Punjabi)
        return "pa", 1.0

    hindi_words = {"hai", "ho", "ka", "ki", "kya", "tum", "aap", "mera", "tera", "nahi", "theek", "acha"}
    punjabi_words = {
        "tusi", "kidda", "kida", "kidaan", "paji", "kar", "karo", "mainu",
        "kirpa", "haan", "nahi", "veere", "chalo", "theek", "mera", "tuhada", "paaji"
    }
    words = set(re.findall(r"\b[a-zA-Z']+\b", text))
    if words & punjabi_words:
        return "pa", 0.95
    if words & hindi_words:
        return "hi", 0.95

    wrong_as_punjabi = {"pt", "ur", "nn", "ro", "gl", "mr", "ne", "gu", "bn"}
    if whisper_lang in wrong_as_punjabi or ft_lang in wrong_as_punjabi:
        if words & punjabi_words:
            return "pa", 0.97
        else:
            return "hi", 0.90

    # 5️⃣ Default fallback
    if ft_lang not in {"en", "hi", "pa"}:
        ft_lang = "en"
    return ft_lang, float(ft_prob)


def play_audio(text, lang_code):
    if not text.strip():
        print("⚠️ No text to speak.")
        return
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            sound = AudioSegment.from_file(tmp.name, format="mp3")
            play(sound)
    except Exception as e:
        print("⚠️ Audio playback error:", e)


def record_audio(duration=5, samplerate=16000):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")
    return np.squeeze(audio)

def recognize_with_whisper(audio_data, samplerate=16000):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio_data, samplerate)
        result = model.transcribe(
            tmp.name,
            task="transcribe",
            language=None,
            condition_on_previous_text=False,
            temperature=0.2,
            beam_size=5,
            best_of=3,
            initial_prompt="Audio may contain English, Hindi or Punjabi spoken with an Indian accent."
        )
        return result["text"].strip(), result["language"]


def translate_and_speak(text, whisper_lang):
    if not text:
        print("⚠️ No speech detected. Try again.")
        return
    smart_lang, prob = detect_smart_language(text, whisper_lang)
    print(f"🧠 Smart AI Detected: {smart_lang} (confidence {prob:.2f})")

    dest_lang = "en" if smart_lang in {"hi", "pa"} else "hi"
    try:
        translated = translator.translate(text, dest=dest_lang)
        print(f"🌐 Translated ({dest_lang}): {translated.text}")
        play_audio(translated.text, dest_lang)
    except Exception as e:
        print("⚠️ Translation error:", e)


def bidirectional_talk():
    audio_data = record_audio(duration=5)
    text, whisper_lang = recognize_with_whisper(audio_data)
    print(f"🗣️ Whisper heard ({whisper_lang}): {text}")
    translate_and_speak(text, whisper_lang)

if __name__ == "__main__":
    print("🚀 VoiceFusion Smart AI Translator (Whisper + FastText Hybrid)")
    print("🎧 English ↔ Hindi ↔ Punjabi (Offline, CPU Friendly)")
    print("Press Ctrl+C to exit.\n")
    while True:
        bidirectional_talk()
