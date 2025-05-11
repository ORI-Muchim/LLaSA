import torch
import torchaudio
import os
from scipy.io.wavfile import write
from .speechsr import SynthesizerTrn

class FASR:
    def __init__(self, ckpt_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = {
            "train": {
                "segment_size": 9600
            },
            "data": {
                "hop_length": 320,
                "n_mel_channels": 128
            },
            "model": {
                "resblock": "0",
                "resblock_kernel_sizes": [3,7,11],
                "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
                "upsample_rates": [3],
                "upsample_initial_channel": 32,
                "upsample_kernel_sizes": [3]
            }
        }
        self.model = self._load_model(ckpt_path)


    def _load_model(self, ckpt_path):
        model = SynthesizerTrn(
            self.hps['data']['n_mel_channels'],
            self.hps['train']['segment_size'] // self.hps['data']['hop_length'],
            **self.hps['model']
        ).to(self.device)
        assert os.path.isfile(ckpt_path)
        checkpoint_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)['model']
        model.load_state_dict(checkpoint_dict)
        model.eval()
        return model


    def load_and_preprocess_audio(self, input_path):
        audio, sample_rate = torchaudio.load(input_path)
        audio = audio[:1, :]
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000, resampling_method="kaiser_window")
        return audio


    def super_resolution(self, audio):
        with torch.no_grad():
            converted_audio = self.model(audio.unsqueeze(1).to(self.device))
            converted_audio = converted_audio.squeeze()
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 0.999
            converted_audio = converted_audio.cpu()
        return converted_audio


    def save_audio(self, audio, output_path):
        audio_numpy = (audio.numpy() * 32767.0).astype('int16')
        write(output_path, 48000, audio_numpy)


    def run(self, input_speech, output_path):
        input_audio = self.load_and_preprocess_audio(input_speech)
        output_audio = self.super_resolution(input_audio)
        self.save_audio(output_audio, output_path)