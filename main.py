import os
import uuid
import shutil
import urllib.parse
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from functions import transcribe_audio, generate_llm_response, text_to_speech
from prompts import SYSTEM_PROMPT

app = FastAPI(
    title="AI Voice Assistant",
    description="A voice-to-voice API and web interface.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcription", "X-LLM-Response"]  # Expose headers for client reading
)

import tempfile

TEMP_DIR = os.path.join(tempfile.gettempdir(), "temp_audio")
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "output_audio")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep track of the latest generated file path in-memory
latest_output_file = None

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify server readiness.
    """
    return {"status": "alive", "message": "API is ready and running."}

@app.post("/input")
async def input_audio(file: UploadFile = File(...)):
    """
    Takes an audio file or recording input, processes it through STT -> LLM -> TTS,
    and directly returns the synthesized audio WAV file.
    Transcriptions and response texts are attached via response headers:
    - X-Transcription
    - X-LLM-Response
    """
    global latest_output_file
    
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # Determine file extension (defaulting to .wav)
    file_extension = os.path.splitext(file.filename)[1]
    if not file_extension:
        file_extension = ".wav"
        
    temp_input_id = str(uuid.uuid4())
    temp_input_path = os.path.join(TEMP_DIR, f"input_{temp_input_id}{file_extension}")
    
    try:
        # Save uploaded file
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Step 1: Transcribe audio to text (STT)
        transcription = transcribe_audio(temp_input_path)
        if not transcription:
            raise HTTPException(
                status_code=400, 
                detail="Could not transcribe audio. Make sure the file contains valid speech."
            )
            
        # Step 2: Query the LLM
        llm_response = generate_llm_response(transcription, SYSTEM_PROMPT)
        
        # Step 3: Convert the LLM response to voice (TTS)
        output_filename = f"output_{temp_input_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        text_to_speech(llm_response, output_path)
        
        # Update latest output tracker
        latest_output_file = output_path
        
        # URL encode headers to be safe for HTTP headers transmission (resolves newline and non-ASCII errors)
        safe_transcription = urllib.parse.quote(transcription)
        safe_llm_response = urllib.parse.quote(llm_response)
        
        headers = {
            "X-Transcription": safe_transcription,
            "X-LLM-Response": safe_llm_response,
            "Access-Control-Expose-Headers": "X-Transcription, X-LLM-Response"
        }
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_filename,
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    finally:
        # Clean up temporary input file
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception:
                pass

@app.get("/output")
def output_audio(file: str = Query(None, description="The specific generated audio filename to download.")):
    """
    Outputs the requested audio file or the latest generated response.
    """
    global latest_output_file
    
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
            
    if latest_output_file and os.path.exists(latest_output_file):
        filename = os.path.basename(latest_output_file)
        return FileResponse(latest_output_file, media_type="audio/wav", filename=filename)
        
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
        newest_file_path = os.path.join(OUTPUT_DIR, files[-1])
        return FileResponse(newest_file_path, media_type="audio/wav", filename=files[-1])
        
    raise HTTPException(status_code=404, detail="No audio outputs available.")

@app.get("/", response_class=HTMLResponse)
def index():
    """
    Returns a premium, modern glassmorphic web interface for the voice assistant.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voice AI Assistant</title>
        <style>
            :root {
                --primary: #8a2be2;
                --primary-glow: rgba(138, 43, 226, 0.4);
                --bg: #030303;
                --glass: rgba(20, 20, 20, 0.7);
                --border: rgba(255, 255, 255, 0.08);
                --text: #ffffff;
                --text-secondary: #888888;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-image: 
                    radial-gradient(circle at 10% 20%, rgba(90, 20, 150, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(30, 40, 120, 0.15) 0%, transparent 45%);
                overflow-x: hidden;
            }

            .container {
                max-width: 600px;
                width: 90%;
                background: var(--glass);
                backdrop-filter: blur(20px);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 2.5rem;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
                text-align: center;
                position: relative;
            }

            h1 {
                font-size: 2.2rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #ffffff 30%, #a060ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }

            .subtitle {
                color: var(--text-secondary);
                font-size: 0.95rem;
                margin-bottom: 2.5rem;
            }

            .main-ui {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 2rem;
            }

            .record-btn-container {
                position: relative;
                width: 140px;
                height: 140px;
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .pulse-ring {
                position: absolute;
                width: 100%;
                height: 100%;
                border-radius: 50%;
                background: var(--primary-glow);
                opacity: 0;
                transform: scale(0.9);
            }

            .pulse-ring.active {
                animation: pulse 1.8s infinite cubic-bezier(0.4, 0, 0.6, 1);
            }

            @keyframes pulse {
                0% {
                    transform: scale(0.9);
                    opacity: 0.7;
                }
                100% {
                    transform: scale(1.4);
                    opacity: 0;
                }
            }

            .record-btn {
                position: relative;
                width: 100px;
                height: 100px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--primary) 0%, #4b0082 100%);
                border: none;
                cursor: pointer;
                box-shadow: 0 10px 25px rgba(138, 43, 226, 0.3), inset 0 2px 2px rgba(255,255,255,0.2);
                display: flex;
                justify-content: center;
                align-items: center;
                transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                z-index: 10;
            }

            .record-btn:active {
                transform: scale(0.95);
            }

            .record-btn svg {
                width: 40px;
                height: 40px;
                fill: var(--text);
                transition: transform 0.3s ease;
            }

            .record-btn.recording svg {
                transform: scale(1.1);
            }

            .status {
                font-size: 1.1rem;
                font-weight: 500;
                color: #ffffff;
                transition: color 0.3s ease;
            }

            .status.active {
                color: #ff3b30;
            }

            .status.processing {
                color: #34c759;
            }

            .visualizer {
                display: flex;
                gap: 5px;
                height: 30px;
                align-items: center;
                opacity: 0;
                transition: opacity 0.3s ease;
            }

            .visualizer.active {
                opacity: 1;
            }

            .bar {
                width: 4px;
                height: 10px;
                background-color: var(--primary);
                border-radius: 2px;
                animation: bounce 1s infinite alternate;
            }

            .bar:nth-child(2) { animation-delay: 0.15s; }
            .bar:nth-child(3) { animation-delay: 0.3s; }
            .bar:nth-child(4) { animation-delay: 0.45s; }
            .bar:nth-child(5) { animation-delay: 0.6s; }

            @keyframes bounce {
                0% { height: 8px; }
                100% { height: 28px; }
            }

            .response-section {
                width: 100%;
                border-top: 1px solid var(--border);
                padding-top: 2rem;
                margin-top: 1rem;
                display: none;
                flex-direction: column;
                gap: 1.2rem;
                text-align: left;
            }

            .log-box {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.2rem;
            }

            .log-title {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: var(--text-secondary);
                margin-bottom: 0.5rem;
                font-weight: 600;
            }

            .log-content {
                font-size: 0.95rem;
                line-height: 1.5;
            }

            .audio-container {
                display: flex;
                justify-content: center;
                margin-top: 0.5rem;
            }

            audio {
                width: 100%;
                outline: none;
            }

            .upload-area {
                width: 100%;
                border: 2px dashed var(--border);
                border-radius: 16px;
                padding: 1.5rem;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.5rem;
            }

            .upload-area:hover {
                border-color: var(--primary);
                background: rgba(138, 43, 226, 0.03);
            }

            .upload-area p {
                font-size: 0.85rem;
                color: var(--text-secondary);
            }

            #file-input {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Voice AI Assistant</h1>
            <div class="subtitle">Direct Voice-to-Voice System powered by Groq</div>

            <div class="main-ui">
                <div class="record-btn-container">
                    <div class="pulse-ring" id="pulse-ring"></div>
                    <button class="record-btn" id="record-btn">
                        <svg viewBox="0 0 24 24">
                            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
                        </svg>
                    </button>
                </div>

                <div class="status" id="status-text">Idle</div>

                <div class="visualizer" id="visualizer">
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                </div>

                <div class="upload-area" id="upload-area">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="stroke: var(--text-secondary)">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    <p>Or click to select an audio file to upload</p>
                    <input type="file" id="file-input" accept="audio/*">
                </div>

                <div class="response-section" id="response-section">
                    <div class="log-box">
                        <div class="log-title">You Said</div>
                        <div class="log-content" id="speech-transcription">...</div>
                    </div>
                    <div class="log-box">
                        <div class="log-title">AI Assistant Response</div>
                        <div class="log-content" id="ai-response">...</div>
                    </div>
                    <div class="audio-container">
                        <audio id="audio-player" controls autoplay></audio>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const recordBtn = document.getElementById('record-btn');
            const pulseRing = document.getElementById('pulse-ring');
            const statusText = document.getElementById('status-text');
            const visualizer = document.getElementById('visualizer');
            const responseSection = document.getElementById('response-section');
            const speechTranscription = document.getElementById('speech-transcription');
            const aiResponse = document.getElementById('ai-response');
            const audioPlayer = document.getElementById('audio-player');
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');

            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;

            // Handle file selection
            uploadArea.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    processAudioFile(e.target.files[0]);
                }
            });

            // Handle recording toggles
            recordBtn.addEventListener('click', async () => {
                if (!isRecording) {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        startRecording(stream);
                    } catch (err) {
                        alert("Microphone permission denied. Please allow microphone access to record.");
                    }
                } else {
                    stopRecording();
                }
            });

            function startRecording(stream) {
                isRecording = true;
                audioChunks = [];
                
                // Set recording options
                let options = {};
                if (MediaRecorder.isTypeSupported('audio/webm')) {
                    options = { mimeType: 'audio/webm' };
                } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                    options = { mimeType: 'audio/ogg' };
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    options = { mimeType: 'audio/mp4' };
                }

                mediaRecorder = new MediaRecorder(stream, options);
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const extension = options.mimeType ? options.mimeType.split('/')[1].split(';')[0] : 'webm';
                    const audioBlob = new Blob(audioChunks, { type: options.mimeType || 'audio/webm' });
                    const audioFile = new File([audioBlob], `recording.${extension}`, { type: options.mimeType || 'audio/webm' });
                    
                    // Stop stream tracks
                    stream.getTracks().forEach(track => track.stop());
                    
                    processAudioFile(audioFile);
                };

                mediaRecorder.start();

                recordBtn.classList.add('recording');
                pulseRing.classList.add('active');
                visualizer.classList.add('active');
                statusText.innerText = "Recording...";
                statusText.classList.add('active');
            }

            function stopRecording() {
                isRecording = false;
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                
                recordBtn.classList.remove('recording');
                pulseRing.classList.remove('active');
                visualizer.classList.remove('active');
                statusText.innerText = "Processing...";
                statusText.classList.remove('active');
                statusText.classList.add('processing');
            }

            async function processAudioFile(file) {
                statusText.innerText = "Processing Audio...";
                statusText.classList.add('processing');
                
                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/input', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP Error ${response.status}`);
                    }

                    // Extract transcriptions and responses from custom headers
                    // Headers are URL encoded to prevent UTF-8 errors, so decode them here
                    const rawTranscript = response.headers.get('X-Transcription') || '';
                    const rawLLMResp = response.headers.get('X-LLM-Response') || '';
                    
                    const transcriptionText = decodeURIComponent(rawTranscript);
                    const llmResponseText = decodeURIComponent(rawLLMResp);

                    speechTranscription.innerText = transcriptionText || "Speech could not be parsed.";
                    aiResponse.innerText = llmResponseText || "No response generated.";

                    // Load returned audio file blob directly into audio player
                    const audioBlob = await response.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayer.src = audioUrl;

                    responseSection.style.display = 'flex';
                    statusText.innerText = "Response Ready!";
                    statusText.classList.remove('processing');
                    
                    setTimeout(() => {
                        statusText.innerText = "Idle";
                    }, 3000);

                } catch (err) {
                    console.error(err);
                    statusText.innerText = "Error processing audio";
                    statusText.classList.remove('processing');
                    alert(`Failed to process audio: ${err.message}`);
                }
            }
        </script>
    </body>
    </html>
    """
