import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model
from FastAudioSR import FASR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the language model
llasa_1b = "HKUSTAudio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko"
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)
print("Language model loaded.")

# Set pad_token_id explicitly to avoid warnings
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.eval().to(device)

# Load the audio codec model
codec_model_path = "HKUSTAudio/xcodec2"
codec_model = XCodec2Model.from_pretrained(codec_model_path)
codec_model.eval().to(device)
print("Audio codec model loaded.")

def ids_to_speech_tokens(speech_ids):
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
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

def generate_zero_shot_tts(reference_wav_path, reference_text, target_text, output_path):
    # Load the reference audio
    # Ensure the reference audio is 16kHz
    ref_wav, sr = sf.read(reference_wav_path)
    ref_wav = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)
    
    full_text = f"{reference_text} {target_text}"
    
    with torch.no_grad():
        # Encode the reference audio to get voice characteristics
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
        
        # Decode the speech tokens to audio
        gen_wav = codec_model.decode_code(speech_ids_tensor)
        gen_wav = gen_wav[:, :, ref_wav.shape[1]:]
        
        # Save the generated audio
        sf.write(output_path, gen_wav[0, 0, :].cpu().numpy(), 16000)

# Example usage
reference_wav_path = "mix.wav"
reference_text = "매니저들의 인맥과 노하우를 활용해서 성사시키기 어려운 계약을 따내거나 부득이하게 겹친 스케쥴을 풀기도 하죠."
target_text = "한덕수 후보를 향해서도 후보님도 끝까지 당에 남아 이번 대선에서 함께 해달라고 요청했다."
output_path = "generated.wav"

# Execute the function
generate_zero_shot_tts(
    reference_wav_path=reference_wav_path,
    reference_text=reference_text,
    target_text=target_text,
    output_path=output_path
)

# Upsample the generated audio from 16kHz to 48kHz
fasr = FASR("FastAudioSR/SR48K.pth")
fasr.run('generated.wav', 'generated_48k.wav')

print("TTS generation complete. Output saved to 'generated.wav' and upsampled to 'generated_48k.wav'")
