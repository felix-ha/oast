import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "de"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, cache_dir="data/hf")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, cache_dir="data/hf")


def get_text(path_to_wav: str) -> str:
    speech_array, sampling_rate = librosa.load(path_to_wav, sr=16_000)
    inputs = processor([speech_array], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    return predicted_sentences[0]

if __name__ == "__main__":
    result = get_text("data/result.wav")
    print(result)

