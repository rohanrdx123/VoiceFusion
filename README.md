# 🎙️ VoiceFusion – Real-Time AI Voice Translator for Client Calls

## 🧠 Overview
**VoiceFusion** is an AI-powered real-time voice translation system that bridges language gaps during client meetings or calls (Google Meet, Zoom, Microsoft Teams).  
It enables seamless bilingual conversations — e.g., an English-speaking client and a Hindi-speaking team — with **automatic speech recognition, translation, and speech synthesis**.

---

## 🚀 Key Features
- 🎤 Real-time **speech recognition** using Whisper (OpenAI)
- 🔁 **Bidirectional translation** (English ↔ Hindi )
- 🔊 **Text-to-Speech** playback for translated audio
- 🔇 **Noise suppression & silence detection** (Silero-VAD)
- 🧭 **Automatic language detection** (FastText)
- 🎧 **Custom output device routing** (e.g., AirPods, Speakers)
- 🧑‍🎤 **Voice selection:** Male / Female
- ⚡ **Async streaming pipeline** (Client ↔ Team)
- 🧩 Modular for integration with **Zoom / Meet / Teams**

---

## 🧩 System Architecture
```
      ┌──────────────────────────────┐
      │           CLIENT             │
      │       Speaks in English      │
      └──────────────┬───────────────┘
                     │  (Audio Input)
                     ▼
            🎙️ Whisper ASR  
      (Speech → Text Conversion)
                     │
                     ▼
            🌐 Translation Engine  
      (English → Hindi )
                     │
                     ▼
            🔊 Text-to-Speech (TTS)  
      (Hindi  Audio Output)
                     │
                     ▼
      🎧 TEAM HEARS IN NATIVE LANGUAGE
```

---

## 🧰 Tech Stack
| Component | Library | Purpose |
|------------|----------|----------|
| 🗣 Speech-to-Text | Whisper (OpenAI) | Speech recognition |
| 🌍 Translation | Helsinki-NLP / MarianMT | Text translation |
| 🔊 TTS | gTTS | Speech synthesis |
| 🧭 Language Detection | FastText | Identify spoken language |
| 🔇 Noise Filtering | Silero-VAD | Remove background noise |
| 🎧 Audio Handling | SoundDevice, PyDub | Record/play audio |
| ⚙️ Async Engine | asyncio | Parallel processing |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rohanrdx123/VoiceFusion.git
cd VoiceFusion
```

### 2️⃣ Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
python voice.py
```

---

## 🧑‍💻 Usage
When you run the script:
```bash
python voice.py
```

You’ll be prompted to:
```
Select team language:
Hindi → hi

Choose voice gender (male/female)
Select audio output device for both Client and Team
```

Then speak — VoiceFusion automatically handles:
```
🎧 Listen → 🧠 Transcribe → 🌐 Translate → 🔊 Speak
```

Press **Ctrl + C** anytime to stop the session.

---

## 💬 Example Workflow
| Speaker | Input Speech | Translated Output |
|----------|---------------|-------------------|
| Client | “Good Morning, how are you?” | “सुप्रभात, आप कैसे हैं?” |
| Team | “मैं ठीक हूँ, धन्यवाद।” | “I am fine, thank you.” |

---

## ⚡ Performance Notes
| Mode | Avg. Latency |
|------|---------------|
| 🖥️ CPU | 2–3 sec/phrase |
| ⚡ Faster-Whisper (INT8) | ~1 sec |
| 💻 GPU/Colab | Near real-time |

> 💡 For production, use **Google Speech API** or **DeepL** for faster, more accurate translations.

---

## 🔮 Future Enhancements
- Chrome/Edge **browser extension**
- Real-time **subtitle overlay**
- **Emotion-aware** voice modulation
- Better translation via **IndicTrans2**
- Add more languages (Spanish, French, etc.)
- **Electron desktop app** packaging

---

## 📜 License
Released under the **MIT License** — free to use and modify.

---

**Made with ❤️ by [Rohan Dixit](https://github.com/rohanrdx123)**
