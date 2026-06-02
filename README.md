# Live Mic Inversion Scope

This app runs in two places:

- Locally: Python serves the page and the browser captures your microphone.
- Railway: Railway serves the same page over HTTPS and the browser still captures your microphone.

The server does not need microphone hardware on Railway. Audio samples are captured in the browser and sent to `/api/invert`, where Python returns the exact electrical inverse for visualization.

## Local Run

```bash
pip install -r requirements.txt
python anc_visual_server.py
```

Open the printed URL:

```text
http://127.0.0.1:8765/anc_visual_dashboard.html
```

## Railway

Railway uses `railway.json`:

```bash
python anc_visual_server.py
```

The server listens on `0.0.0.0:$PORT`, which Railway requires for public networking.

Deploy with:

```bash
railway up
```

## Notes

This is an exact waveform inversion visualizer. It does not play anti-noise or perform real acoustic cancellation.
