# AI Voice Electronics Project

A voice-to-voice FastAPI server using the Groq API for lightning-fast inference.

## Features
- **Health Check**: Endpoint to monitor server readiness.
- **Direct Voice-to-Voice Processing**: A POST endpoint that takes an audio recording, transcribes it, queries the LLM, synthesizes the response text, and directly returns the synthesized audio file (`audio/wav`) in the response body.

## Tech Stack
- **Python** (managed by `uv`)
- **FastAPI** & **Uvicorn**
- **Groq API SDK** (Whisper-Large-V3-Turbo, GPT-OSS-120B, and Canopy Labs Orpheus TTS)

## Running Locally

1. Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
VOICE_INPUT_MODEL=whisper-large-v3-turbo
VOICE_OUTPUT_MODEL=canopylabs/orpheus-v1-english
MAIN_LLM=openai/gpt-oss-120b
```

2. Start the server:
```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
