from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import scipy

#TODO: try https://github.com/coqui-ai/TTS
#TODO: try https://github.com/suno-ai/bark

model = VitsModel.from_pretrained("facebook/mms-tts-deu", cache_dir="data/hf")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-deu", cache_dir="data/hf")

def get_speech(text: str, path: str) -> None:
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    numpy_output = output.cpu().numpy()
    scipy.io.wavfile.write(path, rate=model.config.sampling_rate, data=numpy_output.T)


if __name__ == "__main__":
    text = "Auf einem Steinbruch in England haben Forscher eine atemberaubende Entdeckung gemacht: die größte Fundstätte von Dinosaurierspuren des Landes mit rund 200 Abdrücken. Die Wissenschaftler vermuten in der Nähe noch mehr Spuren."
    get_speech(text, "data/result.wave")

