import os
import tempfile
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

def play_audio_cross_platform(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        sound = AudioSegment.from_file(tmp.name, format="mp3")
        play(sound)

def translate_and_speak(src_lang="en-IN", dest_lang="pa"):
    recognizer = sr.Recognizer()
    translator = Translator()

    with sr.Microphone() as source:
        print("\nSpeak something in English...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language=src_lang)
        print(f"You said: {text}")

        translated = translator.translate(text, dest=dest_lang)
        print(f"Translated ({dest_lang}): {translated.text}")

        play_audio_cross_platform(translated.text, dest_lang)
    except sr.UnknownValueError:
        print("Sorry, I couldn’t understand your speech.")
    except Exception as e:
        print("⚠️ Error:", str(e))

if __name__ == "__main__":
    print("TalkSync Voice Translator (English → Hindi/Punjabi)")
    print("Press Ctrl+C to exit.\n")
    while True:
        translate_and_speak()
