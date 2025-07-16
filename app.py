import os
from pathlib import Path
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
from transformers import pipeline
from google.cloud import texttospeech

from image_caption import generate_caption

BASE_DIR = Path(__file__).parent
cache_dir = BASE_DIR / ".hf_cache"
cache_dir.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(BASE_DIR / "gcloud_credentials.json")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = BASE_DIR / 'uploads'
app.config['STATIC_FOLDER'] = BASE_DIR / 'static'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['STATIC_FOLDER'].mkdir(exist_ok=True)

print("Loading image captioning model...")
print("Initializing Google TTS client...")
tts_client = texttospeech.TextToSpeechClient()
print("Google TTS client initialized.")

def generate_speech_file(text):
    if not text or "Sorry," in text:
        return None

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Studio-O")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        filename = f"speech_{uuid.uuid4()}.mp3"
        output_path = app.config['STATIC_FOLDER'] / filename

        with open(output_path, "wb") as out:
            out.write(response.audio_content)

        return f"/static/{filename}"
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    image_path = app.config['UPLOAD_FOLDER'] / file.filename
    file.save(image_path)

    caption = generate_caption(image_path)
    audio_url = generate_speech_file(caption)

    os.remove(image_path)

    if not audio_url:
        return jsonify({"error": "Failed to generate audio"}), 500

    return jsonify({
        "caption": caption,
        "audio_url": audio_url
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
