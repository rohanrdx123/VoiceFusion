# import os
# import tempfile
# import speech_recognition as sr
# from googletrans import Translator
# from gtts import gTTS
# from pydub import AudioSegment
# from pydub.playback import play
#
# def play_audio_cross_platform(text, lang_code):
#     tts = gTTS(text=text, lang=lang_code)
#     with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
#         tts.save(tmp.name)
#         sound = AudioSegment.from_file(tmp.name, format="mp3")
#         play(sound)
#
# def translate_and_speak(src_lang="en-IN", dest_lang="pa"):
#     recognizer = sr.Recognizer()
#     translator = Translator()
#
#     with sr.Microphone() as source:
#         print("\n🎙️ Speak something in English...")
#         recognizer.adjust_for_ambient_noise(source, duration=0.5)
#         audio = recognizer.listen(source)
#
#     try:
#         text = recognizer.recognize_google(audio, language=src_lang)
#         print(f"📝 You said: {text}")
#
#         translated = translator.translate(text, dest=dest_lang)
#         print(f"🌐 Translated ({dest_lang}): {translated.text}")
#
#         play_audio_cross_platform(translated.text, dest_lang)
#     except sr.UnknownValueError:
#         print("❌ Sorry, I couldn’t understand your speech.")
#     except Exception as e:
#         print("⚠️ Error:", str(e))
#
# if __name__ == "__main__":
#     print("🚀 TalkSync Voice Translator (English → Hindi/Punjabi)")
#     print("Press Ctrl+C to exit.\n")
#     while True:
#         translate_and_speak()
# ---------------------------------------------LANGUAGE DETECT----------------------------------------------
import os
import re
import tempfile
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


def detect_indian_language(text):

    text_lower = text.lower()

    devanagari_pattern = re.compile(r'[\u0900-\u097F]+')   # Hindi script
    gurmukhi_pattern   = re.compile(r'[\u0A00-\u0A7F]+')   # Punjabi script

    hindi_words = ["hai", "ho", "ka", "ki", "kya", "tum", "aap",
                   "mera", "tera", "nahi", "theek", "acha", "sab"]
    punjabi_words = ["ki", "tusi", "tuhada", "mera", "kirpa", "ho",
                     "haan", "nahi", "kidda", "kaim", "kidaan", "kimi"]

    if devanagari_pattern.search(text):
        return "hi"
    if gurmukhi_pattern.search(text):
        return "pa"

    words = text_lower.split()
    if any(w in words for w in punjabi_words):
        return "pa"
    if any(w in words for w in hindi_words):
        return "hi"

    return "en"


# --------------------------- Audio Helpers --------------------------------

def play_audio(text, lang_code):

    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            sound = AudioSegment.from_file(tmp.name, format="mp3")
            play(sound)
    except Exception as e:
        print("⚠️  Audio playback error:", e)


# ---------------------------- Translation ---------------------------------

def translate_text_smart(text, translator):

    detected_lang = detect_indian_language(text)
    print(f"🧠 Smart Detected: {detected_lang}")

    if detected_lang in ["hi", "pa"]:
        dest_lang = "en"
    else:
        dest_lang = "hi"

    translated = translator.translate(text, dest=dest_lang)
    print(f"🌐 Translated ({dest_lang}): {translated.text}")
    play_audio(translated.text, dest_lang)



def bidirectional_talk():
    recognizer = sr.Recognizer()
    translator = Translator()

    with sr.Microphone() as source:
        print("\n🎙️  Speak something (English / Hindi / Punjabi)...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        try:
            text = recognizer.recognize_google(audio, language="en-IN")
        except sr.UnknownValueError:
            text = recognizer.recognize_google(audio, language="hi-IN")

        print(f"🗣️  You said: {text}")
        translate_text_smart(text, translator)

    except Exception as e:
        print("⚠️  Error:", e)


if __name__ == "__main__":
    print("🚀 TalkSync Voice Translator (English ↔ Hindi ↔ Punjabi)")
    print("Press Ctrl+C to exit.\n")

    while True:
        bidirectional_talk()
