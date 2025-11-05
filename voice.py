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
whisper_model = whisper.load_model("small")   # Speech-to-text (multilingual)
lang_model = fasttext.load_model("lid.176.ftz")  # Language ID
translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en").to("cpu").eval()
vad_model = load_silero_vad()
print("✅ All AI models loaded successfully!\n")

SAMPLE_RATE = 16000

# -------------------- AUDIO HELPERS --------------------
def record_chunk(duration=4):
    """Record a short chunk of audio"""
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def remove_silence(audio):
    """Remove silence and background noise using Silero-VAD"""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        wav = read_audio(tmp.name, sampling_rate=SAMPLE_RATE)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLE_RATE)
        if not speech_timestamps:
            return audio
        processed = collect_chunks(speech_timestamps, wav)
        return processed.numpy()

# -------------------- AI CORES --------------------
async def transcribe_audio(audio):
    """Convert speech to text using Whisper"""
    audio = remove_silence(audio)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        result = whisper_model.transcribe(tmp.name)
        return result["text"].strip(), result["language"]

def detect_language(text, whisper_lang=None):
    """Detect dominant language"""
    if not text.strip():
        return "en"
    try:
        label, prob = lang_model.predict(text)
        ft_lang = label[0].replace("__label__", "")
    except Exception:
        ft_lang = "en"
    # Hindi or Punjabi heuristic
    if any(c in text for c in "अआइईउऊएऐओऔकखगघचछजझञटठडढणतथदधनपफबभमयरलवशषसह"):
        return "hi"
    if any(c in text for c in "ਤਸਨਕਪਬਮਲਹਙਜਞਚਛਘਦਧਰਵ"):
        return "pa"
    return whisper_lang or ft_lang

def translate_text(text, src, dest):
    """Translate using Helsinki-NLP model"""
    if not text.strip():
        return ""
    tokenizer = translator_tokenizer
    model = translator_model
    tokenizer.src_lang = src if src in ["en", "hi", "pa"] else "en"
    encoded = tokenizer(text, return_tensors="pt", padding=True).to("cpu")
    generated = model.generate(**encoded)
    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translated

async def speak_text(text, lang):
    """Speak translated text aloud"""
    if not text.strip():
        return
    try:
        tts = gTTS(text=text, lang=lang if lang in ["en", "hi", "pa"] else "en")
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            sound = AudioSegment.from_file(tmp.name, format="mp3")
            play(sound)
    except Exception as e:
        print("⚠️ TTS Error:", e)

# -------------------- USER LOGIC --------------------
async def user_stream(user_label, speak_lang, hear_lang):
    """Continuously listen, translate, and speak"""
    print(f"🎧 {user_label} ready — Speak in [{speak_lang}] → Hear in [{hear_lang}]")
    while True:
        print(f"\n🎙️ {user_label}: Speak now...")
        audio = record_chunk(duration=4)
        text, detected_lang = await transcribe_audio(audio)
        if not text:
            print("🕳️ Silence detected.")
            continue

        print(f"🧠 {user_label} said ({detected_lang}): {text}")
        translated = translate_text(text, speak_lang, hear_lang)
        print(f"🌐 Translated to ({hear_lang}): {translated}")
        await speak_text(translated, hear_lang)

# -------------------- MAIN --------------------
async def main_dual():
    print("🎧 TalkSync — Real-time AI Translator for Calls\n")
    print("Each user can choose what they SPEAK and what they HEAR.\n")

    # User A setup
    a_speak = input("User A — language you SPEAK (en/hi/pa/...): ").strip().lower() or "en"
    a_hear = input("User A — language you HEAR (en/hi/pa/...): ").strip().lower() or "pa"

    # User B setup
    b_speak = input("User B — language you SPEAK (en/hi/pa/...): ").strip().lower() or "pa"
    b_hear = input("User B — language you HEAR (en/hi/pa/...): ").strip().lower() or "en"

    print(f"\n✅ Config Loaded:\nUser A → Speak[{a_speak}] Hear[{a_hear}]\nUser B → Speak[{b_speak}] Hear[{b_hear}]\n")

    await asyncio.gather(
        user_stream("User A", speak_lang=a_speak, hear_lang=b_hear),
        user_stream("User B", speak_lang=b_speak, hear_lang=a_hear)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main_dual())
    except KeyboardInterrupt:
        print("\n🛑 Session Ended.")
