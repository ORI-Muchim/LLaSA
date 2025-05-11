import os
import torch
import gradio as gr
import soundfile as sf
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)
from xcodec2.modeling_xcodec2 import XCodec2Model
from FastAudioSR import FASR
from scipy import signal

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Environment settings to prevent Gradio caching issues
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = "./temp_gradio"

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using device: {device}, tensor type: {torch_dtype}")

# Create directories for temporary files
os.makedirs("temp", exist_ok=True)
os.makedirs("./temp_gradio", exist_ok=True)

# Define sample rate needed for TTS
TARGET_SAMPLE_RATE = 16000  # 16kHz

# Try to load Whisper V3 model
try:
    print("Loading Whisper Large V3 Turbo model...")
    whisper_model_id = "openai/whisper-large-v3-turbo"
    
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    whisper_model.to(device)
    
    whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
    
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("Whisper Large V3 Turbo model loaded successfully.")
    WHISPER_AVAILABLE = True
except Exception as e:
    print(f"Failed to load Whisper model: {str(e)}")
    print("Transcription functionality will be disabled.")
    WHISPER_AVAILABLE = False
    asr_pipe = None

# Load language model
print("Loading language model...")
llasa_1b = "HKUSTAudio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko"
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)

# Set pad_token_id explicitly to prevent warnings
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.eval().to(device)
print("Language model loaded successfully.")

# Load audio codec model
print("Loading audio codec model...")
codec_model_path = "HKUSTAudio/xcodec2"
codec_model = XCodec2Model.from_pretrained(codec_model_path)
codec_model.eval().to(device)
print("Audio codec model loaded successfully.")

# Load FastAudioSR upsampling model
print("Loading FastAudioSR upsampling model...")
fasr = FASR("FastAudioSR/SR48K.pth")
print("FastAudioSR model loaded successfully.")

def ids_to_speech_tokens(speech_ids):
    """Convert voice IDs to tokens"""
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
    """Extract voice IDs from tokens"""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num = int(token_str[4:-2])
                speech_ids.append(num)
            except ValueError:
                print(f"Invalid token format: {token_str}")
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def resample_audio(audio, orig_sr, target_sr):
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio
    
    # Calculate resampling ratio and perform resampling
    resampled_audio = signal.resample_poly(audio, target_sr, orig_sr)
    return resampled_audio

def process_audio_file(audio_path, max_duration=10.0, target_sr=TARGET_SAMPLE_RATE):
    """Process audio file: resampling, mono conversion, length adjustment"""
    # Load audio file
    audio, sr = sf.read(audio_path)
    
    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Check sample rate and resample if needed
    was_resampled = False
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        was_resampled = True
        print(f"Audio resampled from {sr}Hz to {target_sr}Hz.")
    
    # Check maximum length and trim if needed
    max_samples = int(max_duration * target_sr)
    was_trimmed = False
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        was_trimmed = True
        print(f"Audio trimmed to {max_duration} seconds.")
    
    # Save processed audio
    processed_path = os.path.join("temp", "processed_" + os.path.basename(audio_path))
    sf.write(processed_path, audio, target_sr)
    
    return processed_path, was_trimmed, was_resampled

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    if asr_pipe is None:
        raise ValueError("Whisper model is not available.")
    
    try:
        # Use Whisper V3 pipeline for transcription
        result = asr_pipe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_zero_shot_tts(reference_wav_path, reference_text, target_text):
    """Generate zero-shot TTS"""
    # Process reference audio (resampling, mono conversion, length adjustment)
    processed_path, was_trimmed, was_resampled = process_audio_file(reference_wav_path)
    
    # Load reference audio
    ref_wav, sr = sf.read(processed_path)
    ref_wav = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)
    
    full_text = f"{reference_text} {target_text}"
    
    with torch.no_grad():
        # Encode voice characteristics from reference audio
        vq_code_prompt = codec_model.encode_code(input_waveform=ref_wav)
        vq_code_prompt = vq_code_prompt[0, 0, :]
        speech_tokens_prefix = ids_to_speech_tokens(vq_code_prompt)
        
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{full_text}<|TEXT_UNDERSTANDING_END|>"
        
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_tokens_prefix)}
        ]
        
        # Prepare input with chat template
        input_ids = tokenizer.apply_chat_template(
            chat, 
            tokenize=True, 
            return_tensors='pt', 
            continue_final_message=True
        ).to(device)
        
        # Create attention mask to avoid warnings
        attention_mask = torch.ones_like(input_ids)
        
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        
        # Generate speech tokens
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=2048,
            eos_token_id=speech_end_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_p=1,
            temperature=0.8,
        )
        
        # Process generated tokens
        generated_ids = outputs[0][input_ids.shape[1]-len(speech_tokens_prefix):-1]
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_ids = extract_speech_ids(speech_tokens)
        speech_ids_tensor = torch.tensor(speech_ids).unsqueeze(0).unsqueeze(0).to(device)
        
        # Decode speech tokens to audio
        gen_wav = codec_model.decode_code(speech_ids_tensor)
        gen_wav = gen_wav[:, :, ref_wav.shape[1]:]
        
        # Save generated audio
        output_path = os.path.join("temp", "generated.wav")
        sf.write(output_path, gen_wav[0, 0, :].cpu().numpy(), TARGET_SAMPLE_RATE)
        
        # Upsample from 16kHz to 48kHz
        upsampled_path = os.path.join("temp", "generated_48k.wav")
        fasr.run(output_path, upsampled_path)
        
        return upsampled_path, was_trimmed, was_resampled

def process_audio(reference_audio, auto_transcribe):
    """Process audio file and transcribe (optional)"""
    if reference_audio is None:
        return None, "Please upload a reference audio file.", ""
    
    # Process audio (resampling, mono conversion, length adjustment)
    processed_path, was_trimmed, was_resampled = process_audio_file(reference_audio)
    
    # Generate status message
    status_messages = []
    if was_resampled:
        status_messages.append(f"Audio resampled to 16kHz.")
    if was_trimmed:
        status_messages.append(f"Audio trimmed to 10 seconds.")
    
    if not status_messages:
        status_messages.append("Audio is ready.")
    
    status = " ".join(status_messages)
    
    # Process transcription if requested
    transcription = ""
    if auto_transcribe:
        if not WHISPER_AVAILABLE:
            return processed_path, "Whisper model is not available. Transcription is disabled.", ""
        try:
            transcription = transcribe_audio(processed_path)
            status += " Transcription complete."
            return processed_path, status, transcription
        except Exception as e:
            return processed_path, f"{status} Error during transcription: {str(e)}", ""
    else:
        return processed_path, status, ""

def generate_speech(reference_audio, reference_text, target_text):
    """Run TTS pipeline"""
    try:
        # Validate inputs
        if reference_audio is None:
            return None, "Please upload a reference audio file."
        
        if not reference_text.strip():
            return None, "Reference text cannot be empty."
            
        if not target_text.strip():
            return None, "Target text cannot be empty."
        
        # Generate speech
        output_path, was_trimmed, was_resampled = generate_zero_shot_tts(reference_audio, reference_text, target_text)
        
        # Return results
        message_parts = ["Generation complete!"]
        if was_resampled:
            message_parts.append("Audio was resampled to 16kHz.")
        if was_trimmed:
            message_parts.append("Reference audio was trimmed to 10 seconds.")
            
        message = " ".join(message_parts)
        
        return output_path, message
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("""
    # 🎤 Zero-Shot Text-to-Speech Generator
    
    Upload a reference audio sample, provide the text spoken in that sample, and enter new text you want to generate in the same voice.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Reference Audio and Text")
            reference_audio = gr.Audio(label="Upload Reference Audio (max 10 seconds)", type="filepath")
            
            auto_transcribe = gr.Checkbox(
                label="Auto-transcribe with Whisper Large V3 Turbo", 
                value=False,
                interactive=WHISPER_AVAILABLE
            )
            
            if not WHISPER_AVAILABLE:
                gr.Markdown("⚠️ Transcription is disabled: Could not load Whisper model.")
            
            process_btn = gr.Button("Process Audio", variant="primary")
            audio_status = gr.Textbox(label="Status", interactive=False)
            
            reference_text = gr.Textbox(
                label="Reference Text (content of the audio)", 
                lines=3, 
                placeholder="Enter the text spoken in the reference audio..."
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Speech Generation")
            target_text = gr.Textbox(
                label="Target Text (what to generate)", 
                lines=5, 
                placeholder="Enter the text you want to generate in the same voice..."
            )
            generate_btn = gr.Button("Generate Speech", variant="primary")
            generation_status = gr.Textbox(label="Status", interactive=False)
            output_audio = gr.Audio(label="Generated Audio")
            
            # Markdown text moved below output_audio
            gr.Markdown("""
            ### ℹ️ Notes
            - All audio is automatically resampled to 16kHz
            - Reference audio longer than 10 seconds is automatically trimmed
            - The quality of the reference audio affects the voice cloning quality
            - For best results, use clean audio with minimal background noise
            - Whisper Large V3 Turbo model provides automatic transcription functionality
            """)
    
    # Set up event handlers
    process_btn.click(
        fn=process_audio,
        inputs=[reference_audio, auto_transcribe],
        outputs=[reference_audio, audio_status, reference_text]
    )
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[reference_audio, reference_text, target_text],
        outputs=[output_audio, generation_status]
    )

# Run the app
if __name__ == "__main__":
    try:
        # Force share=True to create a public link
        print("Launching Gradio with share=True to create a public link...")
        demo.launch(share=True, server_name="0.0.0.0")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Gradio launch error: {e}")
        # Try alternative launch method if the first fails
        try:
            print("Trying alternative launch method...")
            demo.launch(share=True)
        except Exception as e2:
            print(f"Alternative launch method also failed: {e2}")
        