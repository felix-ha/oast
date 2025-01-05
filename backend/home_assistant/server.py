import random
from loguru import logger
import shutil
import os
from pathlib import Path
from datetime import datetime
import wave
from fastapi import FastAPI, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from stt import get_text
from tts import get_speech
from agents import get_graph


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

graph = get_graph()

def pipeline(path_to_wav: str): 

    try:
        print("processing stt")
        # stt_result = get_text(path_to_wav)
        stt_result = "Erzähle mir einen fun fact aus der geschichte."
        print(f"sst_result: {stt_result}")

        print("invoking graph")
        config = {"configurable": {"thread_id": str(random.randint(0, 1_000))}} # TODO fix state
        state = graph.invoke(input={"messages": [("user", stt_result)]}, config=config)
        print(f"state: {state}")
        print(state)

        llm_answer = state['messages'][-1].content
        print(llm_answer)

        get_speech(llm_answer, "data/out.wav")

        # stt_result = "Hello from pipeline!"
        # llm_answer = "This is the answer"

        return stt_result, llm_answer
    except Exception as e:
        logger.exception("An error occurred")
        raise


@app.post("/upload-wav/")
async def upload_wav_file(audio_file: UploadFile):
    """
    Endpoint to upload and process WAV audio files
    
    Args:
        audio_file (UploadFile): The WAV file to be uploaded
        
    Returns:
        JSONResponse: Information about the processed audio file
    
    Raises:
        HTTPException: If file is not a WAV file or processing fails
    """
    
    # Check if file is a WAV file
    if not audio_file.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="File must be a WAV audio file"
        )
    
    try:
        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{audio_file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
            
        # Read WAV file properties
        with wave.open(file_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            duration = frames / float(frame_rate)

        stt_result, llm_answer = pipeline(file_path)
            
        return JSONResponse(
            content={
                "filename": filename,
                "stt_result": stt_result,
                "llm_answer": llm_answer,
                "file_path": file_path,
                "channels": channels,
                "sample_width_bytes": sample_width,
                "frame_rate_hz": frame_rate,
                "duration_seconds": duration,
                "total_frames": frames
            },
            status_code=200
        )
            
    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )


@app.get("/get-wav")
async def get_wav():
    # Path to the WAV file
    file_path = Path("data/out.wav")
    
    # Check if the file exists
    if not file_path.exists():
        return {"error": "File not found"}
    
    # Read the file content
    file_bytes = file_path.read_bytes()
    
    # Return the WAV file with appropriate headers
    headers = {
        "Content-Disposition": "attachment; filename=out.wav"
    }
    return Response(content=file_bytes, headers=headers, media_type="audio/wav")

