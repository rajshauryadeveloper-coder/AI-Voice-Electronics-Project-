import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
VOICE_INPUT_MODEL = os.environ.get("VOICE_INPUT_MODEL", "whisper-large-v3-turbo")
VOICE_OUTPUT_MODEL = os.environ.get("VOICE_OUTPUT_MODEL", "canopylabs/orpheus-v1-english")
MAIN_LLM = os.environ.get("MAIN_LLM", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

def transcribe_audio(audio_file_path: str) -> str:
    """
    Takes an audio file path, sends it to the Groq Whisper API, and returns the transcribed text.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_file_path}")

    filename = os.path.basename(audio_file_path)
    with open(audio_file_path, "rb") as file:
        response = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model=VOICE_INPUT_MODEL,
        )
    
    if isinstance(response, str):
        return response
    return getattr(response, "text", "")

def generate_llm_response(user_text: str, system_prompt: str) -> str:
    """
    Takes the transcribed text user query and a system prompt, queries the main LLM on Groq,
    and returns the text output.
    """
    response = client.chat.completions.create(
        model=MAIN_LLM,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content

def text_to_speech(text_input: str, output_path: str) -> str:
    """
    Converts LLM's text output query to speech using the Groq audio speech API
    and saves it to the output_path.
    """
    response = client.audio.speech.create(
        model=VOICE_OUTPUT_MODEL,
        voice="troy",  # default voice persona
        input=text_input,
        response_format="wav"
    )
    
    if hasattr(response, "write_to_file"):
        response.write_to_file(output_path)
    else:
        with open(output_path, "wb") as f:
            f.write(response.content)
            
    return output_path
