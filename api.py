import os
import time
import tempfile
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from speculative_whisper import SpeculativeWhisper



app = FastAPI(
    title="Speculative Whisper API",
    version="1.0.0"
)


sw = None


@app.on_event("startup")
async def load_models():
   
    global sw
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    sw = SpeculativeWhisper(
        draft_model="tiny",
        final_model="small",  
        device=device
    )
    print("Models ready")


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Speculative Whisper API is ready",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy",
        "models_loaded": sw is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(...),
    gamma: int = 5,
    max_tokens: int = 500
):
    
    """Transcribe one or more audio files using speculative decoding"""

    
    
    if sw is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet"
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    results = []
    total_start = time.time()
    temp_paths = []

    try:
        
        for file in files:
            
            if not file.filename.endswith(
                (".mp3", ".wav", ".m4a", ".flac", ".ogg")
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file.filename}"
                )

            
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_paths.append(tmp.name)

        
        transcripts = sw.transcribe(
            audio_files=temp_paths,
            gamma=gamma,
            max_tokens=max_tokens
        )

        
        for filename, transcript in zip(
            [f.filename for f in files],
            transcripts
        ):
            results.append({
                "filename": filename,
                "transcript": transcript
            })

        total_time = time.time() - total_start

        return JSONResponse(content={
            "status": "success",
            "total_time": round(total_time, 2),
            "file_count": len(files),
            "results": results
        })

    finally:
        
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post("/transcribe/single")
async def transcribe_single(
    file: UploadFile = File(...),
    gamma: int = 5,
    max_tokens: int = 500
):
   
    """Transcribe a single audio file"""
    
    if sw is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet"
        )

    start = time.time()
    temp_path = None

    try:
        
        if not file.filename.endswith(
            (".mp3", ".wav", ".m4a", ".flac", ".ogg")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.filename}"
            )

        
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        
        transcripts = sw.transcribe(
            audio_files=[temp_path],
            gamma=gamma,
            max_tokens=max_tokens
        )

        elapsed = time.time() - start

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "transcript": transcripts[0],
            "time": round(elapsed, 2)
        })

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )