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


# -------------------- MODEL INITIALIZATION --------------------
print("Loading models...")
whisper_model = whisper.load_model("small")
lang_model = fasttext.load_model("lid.176.ftz")
translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en").to("cpu").eval()
vad_model = load_silero_vad()
print("Models loaded successfully.")

SAMPLE_RATE = 16000


def record_chunk(duration=4):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)


def remove_silence(audio):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        wav = read_audio(tmp.name, sampling_rate=SAMPLE_RATE)
        timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLE_RATE)
        if not timestamps:
            return audio
        processed = collect_chunks(timestamps, wav)
        return processed.numpy()


async def transcribe_audio(audio):
    audio = remove_silence(audio)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        result = whisper_model.transcribe(tmp.name, language=None)
        return result["text"].strip(), result["language"]


def detect_language(text, whisper_lang=None):
    if not text.strip():
        return "en"
    try:
        label, _ = lang_model.predict(text)
        ft_lang = label[0].replace("__label__", "")
    except Exception:
        ft_lang = "en"
    if any(c in text for c in "अआइईउऊएऐओऔकखगघचछजझञटठडढणतथदधनपफबभमयरलवशषसह"):
        return "hi"
    if any(c in text for c in "ਤਸਨਕਪਬਮਲਹਙਜਞਚਛਘਦਧਰਵ"):
        return "pa"
    return whisper_lang or ft_lang


def translate_text(text):
    if not text.strip():
        return ""
    encoded = translator_tokenizer(text, return_tensors="pt", padding=True).to("cpu")
    output = translator_model.generate(**encoded)
    return translator_tokenizer.batch_decode(output, skip_special_tokens=True)[0]


def get_voice_code(lang, gender):
    if lang == "hi":
        return "hi" if gender == "female" else "hi-in"
    if lang == "pa":
        return "pa" if gender == "female" else "pa-in"
    return "en" if gender == "female" else "en-uk"


async def speak_text(text, lang, gender="female", output_device=None):
    if not text.strip():
        return
    try:
        lang_code = get_voice_code(lang, gender)
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            audio = AudioSegment.from_file(tmp.name, format="mp3")
            playback = _play_with_simpleaudio(audio)
            if output_device:
                sd.default.device = output_device
            playback.wait_done()
    except Exception as e:
        print("TTS error:", e)


async def translate_conversation(speaker, src_lang, listener, dest_lang, gender="female", output_device=None):
    print(f"{speaker} → {listener} ({src_lang} → {dest_lang})")
    while True:
        audio = record_chunk(duration=4)
        text, detected_lang = await transcribe_audio(audio)
        if not text:
            continue
        translated = translate_text(text)
        print(f"{speaker}: {text}  =>  {listener}: {translated}")
        await speak_text(translated, dest_lang, gender, output_device)


async def main():
    print("TalkSync: Real-Time AI Translator")

    team_lang = input("Team language (hi for Hindi / pa for Punjabi): ").strip().lower() or "hi"
    team_voice = input("Team voice (male/female): ").strip().lower() or "female"
    client_voice = input("Client voice (male/female): ").strip().lower() or "female"

    print("Available audio devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        print(f"{i}: {d['name']}")
    try:
        client_output = int(input("Device index for Client output: ").strip())
        team_output = int(input("Device index for Team output: ").strip())
    except Exception:
        client_output = None
        team_output = None

    await asyncio.gather(
        translate_conversation("Client", "en", "Team", team_lang, team_voice, team_output),
        translate_conversation("Team", team_lang, "Client", "en", client_voice, client_output)
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Session ended.")
