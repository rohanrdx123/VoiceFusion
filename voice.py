import asyncio
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import fasttext
import whisper
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio, collect_chunks

# -------------------- MODEL LOAD --------------------
print("🚀 Loading AI models... please wait")
whisper_model = whisper.load_model("small")                     # Speech → Text (multilingual)
lang_model = fasttext.load_model("lid.176.ftz")                 # Language Detection
translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en").to("cpu").eval()
vad_model = load_silero_vad()                                   # Noise reduction
print("✅ All AI models loaded successfully!\n")

# -------------------- GLOBAL CONFIG --------------------
SAMPLE_RATE = 16000

def record_chunk(duration=4):
    """Record small duration audio from mic"""
    print("🎙️ Listening...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def remove_silence(audio):
    """Use Silero-VAD to remove silence and noise"""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        wav = read_audio(tmp.name, sampling_rate=SAMPLE_RATE)
        timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLE_RATE)
        if not timestamps:
            return audio
        processed = collect_chunks(timestamps, wav)
        return processed.numpy()

async def transcribe_audio(audio):
    """Whisper → Speech-to-text"""
    audio = remove_silence(audio)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        result = whisper_model.transcribe(tmp.name, language=None)
        return result["text"].strip(), result["language"]

def detect_language(text, whisper_lang=None):
    """Detect dominant language (English/Hindi/Punjabi)"""
    if not text.strip():
        return "en"
    try:
        label, _ = lang_model.predict(text)
        ft_lang = label[0].replace("__label__", "")
    except Exception:
        ft_lang = "en"

    # Simple script-based detection
    if any(c in text for c in "अआइईउऊएऐओऔकखगघचछजझञटठडढणतथदधनपफबभमयरलवशषसह"):
        return "hi"
    if any(c in text for c in "ਤਸਨਕਪਬਮਲਹਙਜਞਚਛਘਦਧਰਵ"):
        return "pa"

    return whisper_lang or ft_lang

def translate_text(text, src, dest):
    """Translate text using Helsinki-NLP model"""
    if not text.strip():
        return ""
    try:
        encoded = translator_tokenizer(text, return_tensors="pt", padding=True).to("cpu")
        output = translator_model.generate(**encoded)
        translated = translator_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return translated
    except Exception as e:
        print("⚠️ Translation error:", e)
        return text

async def speak_text(text, lang):
    """Convert text to speech output"""
    if not text.strip():
        return
    try:
        lang_code = "hi" if lang == "hi" else "pa" if lang == "pa" else "en"
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            sound = AudioSegment.from_file(tmp.name, format="mp3")
            play(sound)
    except Exception as e:
        print("⚠️ TTS error:", e)

# -------------------- MAIN CONVERSATION LOGIC --------------------
async def translate_conversation(speaker, src_lang, listener, dest_lang):
    """One-way translation flow"""
    print(f"\n🎧 {speaker} → {listener} ({src_lang} → {dest_lang})")
    while True:
        audio = record_chunk(duration=4)
        text, detected_lang = await transcribe_audio(audio)
        if not text:
            print("🕳️ Silence detected, skipping.")
            continue

        print(f"🗣️ {speaker} said [{detected_lang}]: {text}")
        translated = translate_text(text, src_lang, dest_lang)
        print(f"🌐 {listener} hears [{dest_lang}]: {translated}")
        await speak_text(translated, dest_lang)

async def main():
    print("\n🎯 TalkSync: Real-Time AI Translator for Client Calls\n")

    # Fixed roles: Client (English), Team (Configurable)
    team_lang = input("Enter team language (hi for Hindi / pa for Punjabi): ").strip().lower() or "hi"

    print(f"\n🧩 Configuration:")
    print(f"Client: Speak[en] ⇄ Hear[{team_lang}]")
    print(f"Team: Speak[{team_lang}] ⇄ Hear[en]\n")

    await asyncio.gather(
        translate_conversation("Client", "en", "Team", team_lang),  # English → Hindi/Punjabi
        translate_conversation("Team", team_lang, "Client", "en")   # Hindi/Punjabi → English
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Session ended.")
