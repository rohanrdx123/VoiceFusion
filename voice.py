import asyncio
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import fasttext
import whisper
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio, collect_chunks


print("⚡ Loading lightweight models...")
whisper_model = whisper.load_model("base")       # Faster
lang_model = fasttext.load_model("lid.176.ftz")
vad_model = load_silero_vad()
print("✅ Models loaded.\n")

SAMPLE_RATE = 16000
MODEL_CACHE = {}  # cache translation models


# ---------- Audio Utils ----------
def record_chunk(duration=6):
    print("🎙️ Listening...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)


def remove_silence(audio):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        wav = read_audio(tmp.name, sampling_rate=SAMPLE_RATE)
        ts = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLE_RATE)
        if not ts:
            return audio
        processed = collect_chunks(ts, wav)
        return processed.numpy()


async def transcribe_audio(audio):
    audio = remove_silence(audio)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        result = whisper_model.transcribe(tmp.name)
        return result["text"].strip(), result["language"]


# ---------- Translation ----------
def get_translation_model(src, dest):
    model_map = {
        ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
        ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
        ("en", "pa"): "Helsinki-NLP/opus-mt-en-pa",
        ("pa", "en"): "Helsinki-NLP/opus-mt-pa-en",
    }
    key = (src, dest)
    if key not in MODEL_CACHE:
        print(f"🔁 Loading translator {src}->{dest} ...")
        model_name = model_map.get(key, "Helsinki-NLP/opus-mt-en-hi")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu").eval()
        MODEL_CACHE[key] = (tokenizer, model)
    return MODEL_CACHE[key]


def translate_text(text, src, dest):
    if not text.strip():
        return ""
    tokenizer, model = get_translation_model(src, dest)
    encoded = tokenizer(text, return_tensors="pt", padding=True).to("cpu")
    output = model.generate(**encoded, max_new_tokens=64)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


# ---------- Speech ----------
async def speak_text(text, lang, output_device=None):
    if not text.strip():
        return
    try:
        tts = gTTS(text=text, lang=lang if lang in ["hi", "pa", "en"] else "en")
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            audio = AudioSegment.from_file(tmp.name, format="mp3")
            if output_device:
                sd.default.device = output_device
            asyncio.create_task(play_async(audio))
    except Exception as e:
        print("TTS error:", e)


async def play_async(audio):
    playback = _play_with_simpleaudio(audio)
    playback.wait_done()


# ---------- Conversation ----------
async def translate_conversation(speaker, src_lang, listener, dest_lang, output_device=None):
    print(f"{speaker} → {listener} ({src_lang} → {dest_lang})")
    while True:
        audio = record_chunk()
        text, _ = await transcribe_audio(audio)
        if not text:
            continue
        translated = translate_text(text, src_lang, dest_lang)
        print(f"{speaker}: {text}  =>  {listener}: {translated}")
        await speak_text(translated, dest_lang, output_device)


# ---------- Main ----------
async def main():
    print("TalkSync (Fast Mode)")
    team_lang = input("Team language (hi for Hindi / pa for Punjabi): ").strip().lower() or "hi"

    devices = sd.query_devices()
    for i, d in enumerate(devices):
        print(f"{i}: {d['name']}")

    try:
        client_output = int(input("Client output device: ").strip())
        team_output = int(input("Team output device: ").strip())
    except Exception:
        client_output = team_output = None

    await asyncio.gather(
        translate_conversation("Client", "en", "Team", team_lang, team_output),
        translate_conversation("Team", team_lang, "Client", "en", client_output)
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Session ended.")
