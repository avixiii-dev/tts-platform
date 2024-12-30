from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
import numpy as np
from TTS.api import TTS
from pathlib import Path
import tempfile
import logging
from typing import Optional
from . import voice_training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS Platform API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Available languages and their models
LANGUAGE_MODELS = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "vi": "tts_models/vi/vivos/vits",
    "fr": "tts_models/fr/mai/tacotron2-DDC",
    "es": "tts_models/es/mai/tacotron2-DDC",
    "de": "tts_models/de/thorsten/tacotron2-DCA",
    "nl": "tts_models/nl/mai/tacotron2-DDC",
}

# TTS instances for each language
tts_instances = {}

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0
    emotion: str = "neutral"
    voice_id: Optional[str] = None  # Add voice_id for custom voices

def load_tts_model(language: str):
    """Load TTS model for a specific language"""
    if language not in LANGUAGE_MODELS:
        raise HTTPException(status_code=400, detail=f"Language {language} not supported")
    
    if language not in tts_instances:
        try:
            logger.info(f"Loading TTS model for language: {language}")
            tts = TTS(model_name=LANGUAGE_MODELS[language], progress_bar=False)
            tts_instances[language] = tts
        except Exception as e:
            logger.error(f"Error loading model for language {language}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading TTS model: {str(e)}")
    
    return tts_instances[language]

def adjust_audio(audio: np.ndarray, speed: float, pitch: float) -> np.ndarray:
    """Adjust audio speed and pitch"""
    try:
        import librosa
        
        # Adjust speed
        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)
        
        # Adjust pitch
        if pitch != 1.0:
            audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=12 * (pitch - 1))
        
        return audio
    except Exception as e:
        logger.error(f"Error adjusting audio: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing audio")

@app.get("/languages")
async def get_available_languages():
    """Get list of available languages"""
    return {"languages": list(LANGUAGE_MODELS.keys())}

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech with customization options"""
    try:
        # Load appropriate TTS model
        tts = load_tts_model(request.language)
        
        # Generate speech
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Generate speech with emotion if supported
            if request.emotion != "neutral" and hasattr(tts, "emotion"):
                wav = tts.tts(
                    text=request.text,
                    emotion=request.emotion,
                    speaker_idx=request.speaker_idx
                )
            else:
                wav = tts.tts(
                    text=request.text,
                    speaker_idx=request.speaker_idx
                )
            
            # Adjust speed and pitch if needed
            if request.speed != 1.0 or request.pitch != 1.0:
                wav = adjust_audio(np.array(wav), request.speed, request.pitch)
            
            # Save to temporary file
            tts.save_wav(wav, temp_file.name)
            
            # Return audio file
            return FileResponse(
                temp_file.name,
                media_type="audio/wav",
                filename=f"tts_output_{hash(request.text)}.wav"
            )
            
    except Exception as e:
        logger.error(f"Error in TTS processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include voice training router
app.include_router(voice_training.router, prefix="/api/v1", tags=["voice-training"])

# Update health check to include custom voices
@app.get("/health")
async def health_check():
    """Check if the service is running and get system info."""
    custom_voices = len(list(Path("app/models/custom_voices").glob("*.json")))
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(tts_instances.keys()),
        "available_languages": list(LANGUAGE_MODELS.keys()),
        "custom_voices": custom_voices
    }
