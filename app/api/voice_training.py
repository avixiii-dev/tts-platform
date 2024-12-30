from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import uuid
import torch
import torchaudio
from typing import List, Optional
import json
import numpy as np
from pathlib import Path

router = APIRouter()

# Constants
UPLOAD_DIR = Path("app/data/voice_samples")
MODELS_DIR = Path("app/models/custom_voices")
SAMPLE_RATE = 22050
MIN_SAMPLES = 5
MAX_SAMPLES = 20
MAX_DURATION = 15  # seconds

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class TrainingStatus(BaseModel):
    voice_id: str
    status: str  # 'collecting', 'training', 'completed', 'failed'
    num_samples: int
    message: Optional[str] = None

class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    language: str
    gender: str
    description: Optional[str] = None

# In-memory storage for training status
training_status = {}

def validate_audio(file_path: Path) -> bool:
    """Validate audio file duration and quality."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        duration = waveform.shape[1] / sample_rate
        
        if duration > MAX_DURATION:
            return False
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        return True
    except Exception:
        return False

async def process_training(voice_id: str, language: str):
    """Background task for processing voice training."""
    try:
        voice_dir = UPLOAD_DIR / voice_id
        samples = list(voice_dir.glob("*.wav"))
        
        if len(samples) < MIN_SAMPLES:
            training_status[voice_id] = TrainingStatus(
                voice_id=voice_id,
                status="failed",
                num_samples=len(samples),
                message=f"Not enough samples. Minimum required: {MIN_SAMPLES}"
            )
            return
            
        # TODO: Implement actual voice training logic here
        # This is a placeholder for the actual training process
        # You would typically:
        # 1. Preprocess audio samples
        # 2. Extract features
        # 3. Train the voice model
        # 4. Save the trained model
        
        # Simulate training process
        import time
        time.sleep(5)  # Simulate training time
        
        # Save model metadata
        model_info = {
            "voice_id": voice_id,
            "language": language,
            "num_samples": len(samples),
            "sample_rate": SAMPLE_RATE,
            "created_at": str(time.time())
        }
        
        with open(MODELS_DIR / f"{voice_id}_metadata.json", "w") as f:
            json.dump(model_info, f)
            
        training_status[voice_id] = TrainingStatus(
            voice_id=voice_id,
            status="completed",
            num_samples=len(samples),
            message="Training completed successfully"
        )
        
    except Exception as e:
        training_status[voice_id] = TrainingStatus(
            voice_id=voice_id,
            status="failed",
            num_samples=0,
            message=str(e)
        )

@router.post("/voices/create")
async def create_voice(voice_info: VoiceInfo):
    """Create a new voice profile."""
    voice_dir = UPLOAD_DIR / voice_info.voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize training status
    training_status[voice_info.voice_id] = TrainingStatus(
        voice_id=voice_info.voice_id,
        status="collecting",
        num_samples=0
    )
    
    # Save voice info
    with open(voice_dir / "info.json", "w") as f:
        json.dump(voice_info.dict(), f)
        
    return voice_info

@router.post("/voices/{voice_id}/upload")
async def upload_sample(
    voice_id: str,
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a voice sample for training."""
    voice_dir = UPLOAD_DIR / voice_id
    
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail="Voice profile not found")
        
    # Check number of existing samples
    existing_samples = len(list(voice_dir.glob("*.wav")))
    if existing_samples >= MAX_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum number of samples ({MAX_SAMPLES}) reached"
        )
        
    # Save the uploaded file
    file_path = voice_dir / f"sample_{existing_samples + 1}.wav"
    with open(file_path, "wb") as f:
        f.write(await audio_file.read())
        
    # Validate audio file
    if not validate_audio(file_path):
        os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio file. Must be WAV format, max {MAX_DURATION} seconds"
        )
        
    # Update training status
    status = training_status.get(voice_id, TrainingStatus(
        voice_id=voice_id,
        status="collecting",
        num_samples=0
    ))
    status.num_samples = existing_samples + 1
    training_status[voice_id] = status
    
    # Start training if we have enough samples
    if status.num_samples >= MIN_SAMPLES:
        # Load voice info to get language
        with open(voice_dir / "info.json", "r") as f:
            voice_info = VoiceInfo(**json.load(f))
            
        status.status = "training"
        training_status[voice_id] = status
        
        if background_tasks:
            background_tasks.add_task(
                process_training,
                voice_id,
                voice_info.language
            )
    
    return status

@router.get("/voices/{voice_id}/status")
async def get_training_status(voice_id: str):
    """Get the current status of voice training."""
    if voice_id not in training_status:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return training_status[voice_id]

@router.get("/voices")
async def list_voices():
    """List all trained voice profiles."""
    voices = []
    for voice_dir in UPLOAD_DIR.iterdir():
        if voice_dir.is_dir():
            info_file = voice_dir / "info.json"
            if info_file.exists():
                with open(info_file, "r") as f:
                    voice_info = json.load(f)
                    status_obj = training_status.get(voice_info["voice_id"])
                    status = status_obj.status if status_obj else "unknown"
                    voice_info["status"] = status
                    voices.append(voice_info)
    return voices
