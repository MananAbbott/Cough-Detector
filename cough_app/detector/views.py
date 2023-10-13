from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from tempfile import NamedTemporaryFile
import subprocess
import joblib

# Create your views here.
def index(request):
    return render(request, 'home.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        aud = request.FILES['audio']

        with NamedTemporaryFile(delete=True, suffix=".webm") as tmp_audio:
            # Write the uploaded audio to the temporary file
            for chunk in aud.chunks():
                tmp_audio.write(chunk)

            # Convert the audio to WAV format using FFmpeg
            with NamedTemporaryFile(delete=True, suffix=".wav") as tmp_wav:
                subprocess.run(["ffmpeg", "-i", tmp_audio.name, tmp_wav.name])
                
                # Read the audio data using soundfile
                audio_file, sample_rate = sf.read(tmp_wav.name)

        if len(audio_file.shape) > 1:
            audio_file = audio_file[:, 0]

        mel_spectrogram = librosa.feature.melspectrogram(y=audio_file, sr=sample_rate, n_mels=128)
        averaged_features = np.mean(mel_spectrogram, axis=1)
        mel_spectrogram_dB = librosa.power_to_db(averaged_features)
        mel_spectrogram_dB =  mel_spectrogram_dB.reshape(-1,1)
        mel_spectrogram_dB = np.hstack((mel_spectrogram_dB.T,[[1]]))

        with open('/Users/manan/Desktop/Grad Classes/CS689/HW02/cough_app/detector/templates/cough_detector.pkl', 'rb') as f:
            theta = joblib.load(f)

        pred = np.sign(mel_spectrogram_dB@theta)
        print(pred[0][0])
        if pred[0][0] == 1:
            response = "Cough Detected!"
        else:
            response = "No Cough Detected!"
        return JsonResponse({'response': response})
