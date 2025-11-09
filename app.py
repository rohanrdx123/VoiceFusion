import streamlit as st
import asyncio
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import fasttext
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio, collect_chunks

# ----------------------------------------------------------
# INITIALIZE MODELS (cached so they load once)
# ----------------------------------------------------------
@st.cache_resource
def load_core_models():
    whisper_model = whisper.load_model("small")
    lang_model = fasttext.load_model("lid.176.ftz")
    vad_model = load_silero_vad()
    return whisper_model, lang_model, vad_model

@st.cache_resource
def load_translation_model(src, dest):
    """Load MarianMT model only once per direction"""
    model_name = f"Helsinki-NLP/opus-mt-{src}-{dest}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu").eval()
    return tokenizer, translator

whisper_model, lang_model, vad_model = load_core_models()
SAMPLE_RATE = 16000

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def record_audio(duration=3):
    """Record from microphone"""
    st.info("🎙️ Speak now...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def remove_silence(audio):
    """Trim silent parts"""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        wav = read_audio(tmp.name, sampling_rate=SAMPLE_RATE)
        timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLE_RATE)
        if not timestamps:
            return audio
        processed = collect_chunks(timestamps, wav)
        return processed.numpy()

async def transcribe_audio(audio):
    """Speech → Text"""
    cleaned = remove_silence(audio)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        sf.write(tmp.name, cleaned, SAMPLE_RATE)
        result = whisper_model.transcribe(tmp.name)
        return result["text"].strip()

def translate_text(text, src="en", dest="hi"):
    """Cached fast translation"""
    tokenizer, translator = load_translation_model(src, dest)
    encoded = tokenizer(text, return_tensors="pt", padding=True)
    output = translator.generate(**encoded)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

def speak_text(text, lang="hi"):
    """Text → Speech"""
    if not text.strip():
        return
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        audio = AudioSegment.from_file(tmp.name, format="mp3")
        play(audio)

# ----------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------
st.set_page_config(page_title="VoiceFusion – Real-Time Translator", page_icon="🎧")
st.title("🎧 VoiceFusion – Real-Time AI Translator")

st.sidebar.header("⚙️ Settings")
lang_choice = st.sidebar.selectbox("Select Team Language", ["hi (Hindi)", "pa (Punjabi)"])
voice_choice = st.sidebar.radio("Voice", ["Female", "Male"])
duration = st.sidebar.slider("🎙️ Recording Duration (sec)", 2, 6, 3)

st.markdown("""
### 💬 How it works:
Click any button → Speak → It listens, translates, and plays the translated voice instantly.
""")

# Maintain session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns(2)
target_lang = "hi" if "hi" in lang_choice else "pa"

# ----------------------------------------------------------
# CLIENT → TEAM
# ----------------------------------------------------------
if col1.button("🎙️ Client → Team (English → Native)"):
    with st.spinner("Listening..."):
        audio = record_audio(duration)
        text = asyncio.run(transcribe_audio(audio))
        translated = translate_text(text, "en", target_lang)
        st.session_state.chat_history.append(("Client", text, translated))
        speak_text(translated, lang=target_lang)

# ----------------------------------------------------------
# TEAM → CLIENT
# ----------------------------------------------------------
if col2.button("🎙️ Team → Client (Native → English)"):
    with st.spinner("Listening..."):
        audio = record_audio(duration)
        text = asyncio.run(transcribe_audio(audio))
        translated = translate_text(text, target_lang, "en")
        st.session_state.chat_history.append(("Team", text, translated))
        speak_text(translated, lang="en")

# ----------------------------------------------------------
# DISPLAY CHAT HISTORY
# ----------------------------------------------------------
st.markdown("## 🗣 Conversation Log")
for speaker, original, translated in st.session_state.chat_history:
    if speaker == "Client":
        st.markdown(f"🧑‍💼 **Client said:** {original}")
        st.markdown(f"➡️ **Translated ({target_lang}):** {translated}")
    else:
        st.markdown(f"👩‍💻 **Team said:** {original}")
        st.markdown(f"➡️ **Translated (English):** {translated}")
    st.markdown("---")

st.caption("Developed by **Rohan Dixit** | VoiceFusion (TalkSync Demo) 🚀")
