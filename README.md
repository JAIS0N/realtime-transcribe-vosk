# Real‑Time Voice Transcription with Vosk

Offline, low‑latency speech‑to‑text for desktop, server, and edge devices.

## Why Vosk?

* **Offline / on‑device**: No network calls; great for privacy and kiosks.
* **Cross‑platform**: Linux, macOS, Windows, ARM (Raspberry Pi, Jetson), Android, iOS.
* **Real‑time**: Streams audio and yields partial (interim) results with final segments.
* **Multilingual**: Community models for many languages.

---

## Quick Start (Python, microphone → console)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install vosk sounddevice

# 1) Download and unzip a Vosk model (e.g., small English) into ./models/en
#    See Vosk model zoo; place the folder path in MODEL_PATH below.
```

```python
# file: mic_realtime.py
import queue, sys, json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = "models/en"  # folder containing 'conf', 'am', etc.
SAMPLE_RATE = 16000

q = queue.Queue()

def _callback(indata, frames, t, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

if __name__ == "__main__":
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)  # word‑level timestamps

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,       # ~0.25s chunks; lower = lower latency
        dtype='int16',
        channels=1,
        callback=_callback,
    ):
        print("🎙️  Listening… Ctrl+C to stop")
        try:
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    # Finalized segment
                    result = json.loads(rec.Result())
                    print("\nFINAL:", result.get("text", ""))
                else:
                    # Partial hypothesis (interim)
                    partial = json.loads(rec.PartialResult())
                    print("\rpartial:", partial.get("partial", ""), end="")
        except KeyboardInterrupt:
            print("\n", json.loads(rec.FinalResult()))
```

Run it:

```bash
python mic_realtime.py
```

> If you hit PortAudio errors on Linux, install the system lib first, e.g. `sudo apt-get install libportaudio2`.

---

## Minimal WebSocket Server (Python → FastAPI)

Serve Vosk over WebSocket so browsers/devices can stream raw PCM.

```bash
pip install fastapi uvicorn
```

```python
# file: server_ws.py
import json
from fastapi import FastAPI, WebSocket
from vosk import Model, KaldiRecognizer

MODEL_PATH = "models/en"
SAMPLE_RATE = 16000

app = FastAPI()
model = Model(MODEL_PATH)

@app.websocket("/ws")
async def ws_recognize(ws: WebSocket):
    await ws.accept()
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)
    try:
        while True:
            # Expect 16kHz mono, 16‑bit little‑endian PCM bytes per message
            data = await ws.receive_bytes()
            if rec.AcceptWaveform(data):
                await ws.send_text(rec.Result())   # finalized JSON
            else:
                await ws.send_text(rec.PartialResult())  # partial JSON
    except Exception:
        await ws.close()
```

Run the server:

```bash
uvicorn server_ws:app --host 0.0.0.0 --port 8000
```

### Tiny Browser Client (WebAudio → PCM → WS)

> Sends 16‑bit PCM frames (~20–50 ms) to the server above.

```html
<script>
(async () => {
  const ws = new WebSocket("ws://localhost:8000/ws");
  ws.onmessage = (e) => console.log(JSON.parse(e.data));

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext({ sampleRate: 16000 });
  const src = ctx.createMediaStreamSource(stream);
  await ctx.audioWorklet.addModule('pcm-worklet.js');
  const node = new AudioWorkletNode(ctx, 'pcm-worklet');

  node.port.onmessage = (e) => ws.readyState === 1 && ws.send(e.data);
  src.connect(node);  // no need to connect to destination
})();
</script>
```

```js
// file: pcm-worklet.js
class PCMWorklet extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    // Float32 [-1,1] → Int16LE bytes
    const ab = new ArrayBuffer(input.length * 2);
    const view = new DataView(ab);
    for (let i = 0; i < input.length; i++) {
      let s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    this.port.postMessage(ab);
    return true;
  }
}
registerProcessor('pcm-worklet', PCMWorklet);
```

---

## Output JSON shape

Vosk returns JSON strings:

* **Partial** (interim): `{"partial": "hello wor"}`
* **Final** segment:

  ```json
  {
    "result": [
      {"conf": 0.93, "start": 0.12, "end": 0.48, "word": "hello"},
      {"conf": 0.88, "start": 0.49, "end": 0.93, "word": "world"}
    ],
    "text": "hello world"
  }
  ```

---

## Accuracy & Latency Tips

* **Model size vs speed**: Small models are fast but less accurate; large models are better but heavier. Start with a small model; switch to a full model for production.
* **Chunking**: `blocksize` 4000–8000 bytes (≈0.125–0.25 s) is a good real‑time trade‑off.
* **Sample rate**: 16 kHz mono `int16` is standard for most Vosk models.
* **VAD (optional)**: Use `webrtcvad` to drop non‑speech; reduces CPU and false inserts.
* **Custom vocabulary / grammar**: Constrain recognition with a small phrase list:

  ```python
  import json
  grammar = json.dumps(["yes", "no", "start the job", "stop the job"])
  rec = KaldiRecognizer(model, 16000, grammar)
  ```
* **Punctuation & casing**: Add a lightweight post‑processor (e.g., a punctuation restoration model) if your base model lacks punctuation.
* **Diarization / speaker tags**: Optional speaker embeddings are supported when you load a speaker model and call `rec.SetSpkModel(...)`.

---

## Saving subtitles (SRT) from results

```python
# file: results_to_srt.py
import json, datetime

def to_srt(results):
    """results: list of finalized JSON strings from Vosk"""
    def fmt(t):
        ms = int((t % 1) * 1000)
        s = int(t) % 60
        m = (int(t) // 60) % 60
        h = int(t) // 3600
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    out, idx = [], 1
    for r in results:
        j = json.loads(r)
        if not j.get('result'): continue
        start = j['result'][0]['start']
        end   = j['result'][-1]['end']
        text  = j.get('text', '')
        out.append(f"{idx}\n{fmt(start)} --> {fmt(end)}\n{text}\n")
        idx += 1
    return "\n".join(out)
```

---

## Directory Layout

```
.
├── mic_realtime.py            # microphone → console demo
├── server_ws.py               # WebSocket server exposing Vosk
├── pcm-worklet.js             # Browser AudioWorklet: Float32 → Int16 PCM
├── results_to_srt.py          # Convert finalized results to SRT
└── models/
    └── en/                    # unpacked Vosk model directory
```

---

## Common Issues & Fixes

* **PortAudio not found**: Install the system lib (e.g., `apt-get install libportaudio2` on Debian/Ubuntu, `brew install portaudio` on macOS). If `sounddevice` is troublesome, try `pip install pyaudio` instead.
* **Wrong audio format**: Ensure **16 kHz, mono, 16‑bit little‑endian PCM** when streaming to the recognizer.
* **High latency**: Reduce `blocksize`, prefer smaller models, disable unnecessary post‑processing.
* **CPU usage on ARM**: Use small models; compile with platform optimizations; consider running recognition in a separate process.

---

## Security & Privacy

* Runs entirely offline; no audio leaves the device unless you send it.
* If you expose a WS endpoint, add TLS and authentication. Log only what you need.

---

## Use Cases & Patterns

* **Live captions** for meetings/classrooms (laptop or kiosk).
* **Accessibility** overlays: on‑screen text for hearing assistance.
* **Contact centers (edge)**: on‑prem transcription where cloud is restricted.
* **Field‑service / warehouses**: hands‑free commands with constrained grammar.
* **Robotics / voice HMI**: wake words + short commands fully offline.
* **Transcribe videos** into searchable text and **generate subtitles (SRT/VTT)**.
* **Kiosks & signage**: multi‑language guest interactions without internet.
* **Smart devices / IoT**: embedded boards (Raspberry Pi, Jetson) with small models.

---

## Next Steps

* Swap in a larger language‑appropriate model for better accuracy.
* Add punctuation restoration and optional diarization.
* Persist timestamps to a store (e.g., SQLite) and build a simple transcript UI.

---

## License

This README and snippets are provided under the MIT license. Vosk and its models have their own licenses—review them before distribution.
