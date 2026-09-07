# Real-Time Voice Assistant (Offline, Audio-In / Audio-Out)

A conversational AI assistant that listens through the microphone,
transcribes speech, generates a response, and speaks it back —
built for the **AI Engineering Intern** take-home task.

## Architecture

```
Mic → VAD (turn detection) → Whisper (STT) → Ollama LLM (streaming)
    → Piper (TTS, streamed sentence-by-sentence) → Speakers
                     |
                     └── Fallback watchdog: plays a natural filler
                         phrase if no audio has started within
                         ~0.9s, so the user is never left waiting
                         in silence.
```

Everything runs **fully offline** on your machine — no API keys, no
internet dependency at inference time (matches the brief's preference
for an offline implementation).

## 1. Setup

### Install system dependencies
```bash
# Ollama (runs the local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# Python deps
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Download the Piper voice model
```bash
mkdir -p models
cd models
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
```

### Pre-generate fallback filler audio (do this once)
```bash
python fallback.py
```
This synthesizes the filler phrases in `fallback.py` and caches them
as `.wav` files in `fillers/`, so there's zero synthesis delay when
the fallback is triggered at runtime.

## 2. Run it
```bash
python main.py
```
Speak after "Listening..." appears. The assistant detects when you
stop talking (via VAD, not a fixed timer), transcribes, responds,
and speaks back — logging latency for every turn to
`latency_log.csv`.

## 3. Tuning for the 2-second target

Latency is logged per-turn (`capture`, `stt`, `time_to_first_token`,
`time_to_first_audio`, `total_turn`). If you're missing the budget:

- Drop `WHISPER_MODEL_SIZE` to `"base"` or `"tiny"` in `config.py`
- Drop `OLLAMA_MODEL` to a smaller model (e.g. `qwen2.5:1.5b`)
- Shorten `SYSTEM_PROMPT` to force shorter replies
- Lower `SILENCE_TIMEOUT_MS` so turn-taking feels snappier
- If you have a GPU, set `WHISPER_DEVICE = "cuda"` and
  `WHISPER_COMPUTE_TYPE = "float16"`

The `time_to_first_audio` metric is the one that matters most for
perceived responsiveness — that's when the user actually hears
something, which is what "within 2 seconds" should be measured against.

## 4. What makes the fallback good (not just a spinner)

Per the brief: *"keep the user engaged rather than leaving them
waiting."* Instead of dead air or an error message, a background
watchdog thread plays a short, pre-cached, natural-sounding filler
("Let me think about that for a second...") if the real response
hasn't started by ~0.9s in. Because the filler clips are pre-generated
offline, playing one costs no extra latency of its own.

## 5. Submission checklist (per the internship brief)

- [ ] Upload this whole project + a short screen-recorded demo to Google Drive
- [ ] Create a PDF listing all Drive links
- [ ] **Disclose AI usage explicitly**, e.g.:
  > "AI assistance (Claude) was used to scaffold the initial project
  > architecture, pipeline wiring, and README documentation. The
  > model selection, latency tuning, and testing were done manually."
  > (Adjust this line to reflect what you actually did — be specific and honest.)
- [ ] Double-check every Drive link is set to "Anyone with the link can view" before submitting
- [ ] Include your `latency_log.csv` or a summary of your measured latencies as evidence you hit the target

## Extending it further (optional, for a stronger submission)

- Add barge-in support (let the user interrupt the assistant mid-reply)
- Swap Piper for a voice-cloned model if you want a more distinctive voice
- Add a simple web UI (e.g. with FastAPI + WebSockets) instead of a CLI loop, if the brief allows a demo interface
