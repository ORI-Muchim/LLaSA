# LLaSA: Scaling Train-time and Inference-time Compute for LLaMA-based Speech Synthesis

LLaSA is a zero-shot text-to-speech system that extracts voice characteristics from a reference audio sample and generates new text in the same voice. With just a 10-second audio clip, you can clone any voice with high quality.

## Features

- **Zero-Shot Voice Cloning**: Clone any voice with a single reference audio
- **Automatic Speech Recognition (ASR)**: High-quality transcription using Whisper Large V3 Turbo
- **High-Quality Output**: Automatic upsampling from 16kHz to 48kHz
- **User-Friendly Interface**: Web-based UI for easy use
- **Automatic Processing**: Supports all audio formats and sample rates
- **Multilingual Support**: Works with English, Korean, Japanese, Chinese, and more

## Installation

### Requirements

- Python 3.9
- PyTorch
- CUDA-compatible GPU recommended (8GB+ VRAM)

### Setup

```bash
# Clone the repository
git clone https://github.com/ORI-Muchim/LLaSA.git
cd LLaSA

# Install dependencies
pip install -r requirements.txt

## Usage

### Web Interface

```bash
python app.py
```

This will launch a Gradio web interface accessible through a browser. The interface provides options to:
1. Upload reference audio (max 10 seconds)
2. Transcribe it automatically (optional)
3. Enter or edit the reference text
4. Enter the target text to generate
5. Generate and play/download the resulting audio

### Command Line

```bash
python main.py
```

## Models

LLaSA integrates several models:
- **LLaSA 1B**: LLaMA-based language model for speech generation
- **XCodec2**: Audio codec model for encoding/decoding
- **Whisper Large V3 Turbo**: OpenAI's latest ASR model
- **FastAudioSR**: Super-resolution model for audio quality enhancement

## Tips for Best Results

- Use high-quality reference audio with minimal background noise
- Reference audio should be at least 5 seconds long
- Speak clearly in the reference audio
- For best results, match the language and speaking style in your target text

## Citation

```
@article{ye2025llasa,
  title={Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis},
  author={Ye, Zhen and Zhu, Xinfa and Chan, Chi-Min and Wang, Xinsheng and Tan, Xu and Lei, Jiahe and Peng, Yi and Liu, Haohe and Jin, Yizhu and Dai, Zheqi and Lin, Hongzhan and Chen, Jianyi and Du, Xingjian and Xue, Liumeng and Chen, Yunlin and Li, Zhifei and Xie, Lei and Kong, Qiuqiang and Guo, Yike and Xue, Wei},
  journal={arXiv preprint arXiv:2502.04128v2},
  year={2025}
}
```

## Reference

[project-elnino/FastAudioSR](https://github.com/project-elnino/FastAudioSR)