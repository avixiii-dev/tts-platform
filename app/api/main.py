from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
from TTS.utils.synthesizer import Synthesizer
from pathlib import Path

app = FastAPI(title="TTS Platform API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Initialize TTS synthesizer
synthesizer = None

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    pitch: float = 1.0
    emotion: str = "neutral"

@app.on_event("startup")
async def startup_event():
    global synthesizer
    try:
        # Initialize with a default model
        # You can replace these with your preferred model
        model_path = "tts_models/en/ljspeech/tacotron2-DDC"
        synthesizer = Synthesizer(
            tts_checkpoint="",  # Will be downloaded automatically
            tts_config_path="",
            vocoder_checkpoint="",
            vocoder_config="",
            use_cuda=torch.cuda.is_available()
        )
    except Exception as e:
        print(f"Error initializing synthesizer: {e}")

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using the specified parameters"""
    try:
        # Generate unique filename for the output
        output_path = MODELS_DIR / f"output_{hash(request.text)}.wav"
        
        # Generate speech
        wav = synthesizer.tts(
            text=request.text,
            speaker_name=None,
            language_name=None
        )
        
        # TODO: Apply speed and pitch modifications
        # This will require additional audio processing
        
        # Save the audio file
        synthesizer.save_wav(wav, output_path)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_voice(audio_file: UploadFile = File(...)):
    """Train a custom voice model using uploaded audio"""
    try:
        # Save uploaded file
        file_path = MODELS_DIR / audio_file.filename
        with file_path.open("wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # TODO: Implement voice training logic
        # This will require setting up a training pipeline
        
        return {"message": "Training started", "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}
