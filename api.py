import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from functions import transcribe_audio, generate_llm_response, text_to_speech
from prompts import SYSTEM_PROMPT

app = FastAPI(
    title="Voice-to-Voice FastAPI Server",
    description="An API with health check, audio input processing, and audio output endpoints.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_audio"
OUTPUT_DIR = "output_audio"

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep track of the latest generated file path in-memory
latest_output_file = None

@app.get("/health")
def health_check():
    """
    1. The first endpoint is a health check just to see whether the API is alive and ready.
    """
    return {"status": "alive", "message": "API is ready and running."}

@app.post("/input")
async def input_audio(file: UploadFile = File(...)):
    """
    2. The second endpoint is an input which takes the input of an audio file or audio recording,
       processes it through the pipeline, generates the output audio, and returns a details summary.
    """
    global latest_output_file
    
    # Check that a file was actually uploaded
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # Determine file extension (defaulting to .wav if unspecified)
    file_extension = os.path.splitext(file.filename)[1]
    if not file_extension:
        file_extension = ".wav"
        
    temp_input_id = str(uuid.uuid4())
    temp_input_path = os.path.join(TEMP_DIR, f"input_{temp_input_id}{file_extension}")
    
    try:
        # Save uploaded file
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Step 1: Transcribe audio to text
        transcription = transcribe_audio(temp_input_path)
        if not transcription:
            raise HTTPException(
                status_code=400, 
                detail="Could not transcribe audio. Make sure the file contains valid speech."
            )
            
        # Step 2: Query the LLM
        llm_response = generate_llm_response(transcription, SYSTEM_PROMPT)
        
        # Step 3: Convert the LLM query output to voice
        output_filename = f"output_{temp_input_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        text_to_speech(llm_response, output_path)
        
        # Update our latest output file tracker
        latest_output_file = output_path
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    finally:
        # Clean up the uploaded temporary input file
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception:
                pass

@app.get("/output")
def output_audio(file: str = Query(None, description="The specific generated audio filename to download.")):
    """
    3. The third endpoint is an output which outputs an audio file or audio recording.
       If the 'file' query parameter is provided, returns that specific file.
       Otherwise, returns the latest generated voice output file.
    """
    global latest_output_file
    
    # Return specific file if requested
    if file:
        file_path = os.path.join(OUTPUT_DIR, file)
        # Prevent Directory Traversal vulnerability
        resolved_path = os.path.abspath(file_path)
        expected_dir = os.path.abspath(OUTPUT_DIR)
        if not resolved_path.startswith(expected_dir):
            raise HTTPException(status_code=400, detail="Access denied: Invalid filename.")
            
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/wav", filename=file)
        else:
            raise HTTPException(status_code=404, detail=f"Audio file '{file}' not found.")
            
    # Return latest output if available
    if latest_output_file and os.path.exists(latest_output_file):
        filename = os.path.basename(latest_output_file)
        return FileResponse(latest_output_file, media_type="audio/wav", filename=filename)
        
    # Scan directory for newest file if global tracker is empty (e.g. after server restart)
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
        newest_file_path = os.path.join(OUTPUT_DIR, files[-1])
        return FileResponse(newest_file_path, media_type="audio/wav", filename=files[-1])
        
    raise HTTPException(status_code=404, detail="No audio outputs available.")
