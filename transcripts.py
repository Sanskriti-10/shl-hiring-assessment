import whisper
import os
import json

model = whisper.load_model("base")  # or "small", "medium", etc.

AUDIO_DIR = "./shl-intern-hiring-assessment/Dataset/audios/train"
transcripts = {}

for fname in os.listdir(AUDIO_DIR):
    if fname.endswith(".wav"):
        audio_path = os.path.join(AUDIO_DIR, fname)
        result = model.transcribe(audio_path)
        transcripts[fname] = result['text']

# Save to JSON
with open("train_transcripts.json", "w") as f:
    json.dump(transcripts, f)

#This code is used to convert audio files to transcripts in json format.
