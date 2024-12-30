# TTS Platform

A powerful Text-to-Speech platform using Coqui TTS with customizable voice features.

## Features

- Natural and expressive text-to-speech conversion
- Voice customization (pitch, speed, emotions)
- Custom voice model training
- User-friendly web interface
- Multi-language support

## Tech Stack

- Backend: FastAPI + Coqui TTS
- Frontend: React
- ML: PyTorch + Coqui TTS models

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
cd app/api
uvicorn main:app --reload
```

4. Start the frontend (in a separate terminal):
```bash
cd app/frontend
npm install
npm start
```

## Project Structure

```
tts-platform/
├── app/
│   ├── api/             # FastAPI backend
│   ├── frontend/        # React frontend
│   └── models/          # TTS model storage
├── requirements.txt     # Python dependencies
└── README.md
```

## License

MIT
