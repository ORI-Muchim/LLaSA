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

# Gradio 캐싱 이슈 방지를 위한 환경 설정
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = "./temp_gradio"

# CUDA 사용 가능 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"사용 장치: {device}, 텐서 타입: {torch_dtype}")

# 임시 파일을 위한 디렉토리 생성
os.makedirs("temp", exist_ok=True)
os.makedirs("./temp_gradio", exist_ok=True)

# TTS에 필요한 샘플 레이트 정의
TARGET_SAMPLE_RATE = 16000  # 16kHz

# Whisper V3 모델 로드 시도
try:
    print("Whisper Large V3 Turbo 모델 로드 중...")
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
    
    print("Whisper Large V3 Turbo 모델 로드 완료.")
    WHISPER_AVAILABLE = True
except Exception as e:
    print(f"Whisper 모델 로드 실패: {str(e)}")
    print("전사 기능은 비활성화됩니다.")
    WHISPER_AVAILABLE = False
    asr_pipe = None

# 언어 모델 로드
print("언어 모델 로드 중...")
llasa_1b = "HKUSTAudio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko"
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)

# pad_token_id 명시적 설정 (경고 방지)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.eval().to(device)
print("언어 모델 로드 완료.")

# 오디오 코덱 모델 로드
print("오디오 코덱 모델 로드 중...")
codec_model_path = "HKUSTAudio/xcodec2"
codec_model = XCodec2Model.from_pretrained(codec_model_path)
codec_model.eval().to(device)
print("오디오 코덱 모델 로드 완료.")

# FastAudioSR 업샘플링 모델 로드
print("FastAudioSR 업샘플링 모델 로드 중...")
fasr = FASR("FastAudioSR/SR48K.pth")
print("FastAudioSR 모델 로드 완료.")

def ids_to_speech_tokens(speech_ids):
    """음성 ID를 토큰으로 변환"""
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
    """토큰에서 음성 ID 추출"""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num = int(token_str[4:-2])
                speech_ids.append(num)
            except ValueError:
                print(f"유효하지 않은 토큰 형식: {token_str}")
        else:
            print(f"예상치 못한 토큰: {token_str}")
    return speech_ids

def resample_audio(audio, orig_sr, target_sr):
    """오디오를 target_sr로 리샘플링"""
    if orig_sr == target_sr:
        return audio
    
    # 리샘플링 계수 계산 및 리샘플링 수행
    resampled_audio = signal.resample_poly(audio, target_sr, orig_sr)
    return resampled_audio

def process_audio_file(audio_path, max_duration=10.0, target_sr=TARGET_SAMPLE_RATE):
    """오디오 파일 처리: 리샘플링, 모노 변환, 길이 조정"""
    # 오디오 파일 로드
    audio, sr = sf.read(audio_path)
    
    # 스테레오를 모노로 변환
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # 샘플 레이트 확인 및 리샘플링
    was_resampled = False
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        was_resampled = True
        print(f"오디오를 {sr}Hz에서 {target_sr}Hz로 리샘플링했습니다.")
    
    # 최대 길이 확인 및 잘라내기
    max_samples = int(max_duration * target_sr)
    was_trimmed = False
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        was_trimmed = True
        print(f"오디오를 {max_duration}초로 잘랐습니다.")
    
    # 처리된 오디오 저장
    processed_path = os.path.join("temp", "processed_" + os.path.basename(audio_path))
    sf.write(processed_path, audio, target_sr)
    
    return processed_path, was_trimmed, was_resampled

def transcribe_audio(audio_path):
    """Whisper를 사용하여 오디오 전사"""
    if asr_pipe is None:
        raise ValueError("Whisper 모델을 사용할 수 없습니다.")
    
    try:
        # Whisper V3 파이프라인을 사용한 전사
        result = asr_pipe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"전사 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_zero_shot_tts(reference_wav_path, reference_text, target_text):
    """제로샷 TTS 생성"""
    # 참조 오디오 처리 (리샘플링, 모노 변환, 길이 조정)
    processed_path, was_trimmed, was_resampled = process_audio_file(reference_wav_path)
    
    # 참조 오디오 로드
    ref_wav, sr = sf.read(processed_path)
    ref_wav = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)
    
    full_text = f"{reference_text} {target_text}"
    
    with torch.no_grad():
        # 참조 오디오에서 음성 특성 인코딩
        vq_code_prompt = codec_model.encode_code(input_waveform=ref_wav)
        vq_code_prompt = vq_code_prompt[0, 0, :]
        speech_tokens_prefix = ids_to_speech_tokens(vq_code_prompt)
        
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{full_text}<|TEXT_UNDERSTANDING_END|>"
        
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_tokens_prefix)}
        ]
        
        # 채팅 템플릿으로 입력 준비
        input_ids = tokenizer.apply_chat_template(
            chat, 
            tokenize=True, 
            return_tensors='pt', 
            continue_final_message=True
        ).to(device)
        
        # 경고 방지를 위한 어텐션 마스크 생성
        attention_mask = torch.ones_like(input_ids)
        
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        
        # 음성 토큰 생성
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
        
        # 생성된 토큰 처리
        generated_ids = outputs[0][input_ids.shape[1]-len(speech_tokens_prefix):-1]
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_ids = extract_speech_ids(speech_tokens)
        speech_ids_tensor = torch.tensor(speech_ids).unsqueeze(0).unsqueeze(0).to(device)
        
        # 음성 토큰을 오디오로 디코딩
        gen_wav = codec_model.decode_code(speech_ids_tensor)
        gen_wav = gen_wav[:, :, ref_wav.shape[1]:]
        
        # 생성된 오디오 저장
        output_path = os.path.join("temp", "generated.wav")
        sf.write(output_path, gen_wav[0, 0, :].cpu().numpy(), TARGET_SAMPLE_RATE)
        
        # 16kHz에서 48kHz로 업샘플링
        upsampled_path = os.path.join("temp", "generated_48k.wav")
        fasr.run(output_path, upsampled_path)
        
        return upsampled_path, was_trimmed, was_resampled

def process_audio(reference_audio, auto_transcribe):
    """오디오 파일 처리 및 전사 (선택 사항)"""
    if reference_audio is None:
        return None, "참조 오디오 파일을 업로드해 주세요.", ""
    
    # 오디오 처리 (리샘플링, 모노 변환, 길이 조정)
    processed_path, was_trimmed, was_resampled = process_audio_file(reference_audio)
    
    # 상태 메시지 생성
    status_messages = []
    if was_resampled:
        status_messages.append(f"오디오를 16kHz로 리샘플링했습니다.")
    if was_trimmed:
        status_messages.append(f"오디오를 10초로 잘랐습니다.")
    
    if not status_messages:
        status_messages.append("오디오가 준비되었습니다.")
    
    status = " ".join(status_messages)
    
    # 전사 요청 시 처리
    transcription = ""
    if auto_transcribe:
        if not WHISPER_AVAILABLE:
            return processed_path, "Whisper 모델을 사용할 수 없습니다. 전사 기능이 비활성화되었습니다.", ""
        try:
            transcription = transcribe_audio(processed_path)
            status += " 전사 완료."
            return processed_path, status, transcription
        except Exception as e:
            return processed_path, f"{status} 전사 중 오류 발생: {str(e)}", ""
    else:
        return processed_path, status, ""

def generate_speech(reference_audio, reference_text, target_text):
    """TTS 파이프라인 실행"""
    try:
        # 입력 유효성 검사
        if reference_audio is None:
            return None, "참조 오디오 파일을 업로드해 주세요."
        
        if not reference_text.strip():
            return None, "참조 텍스트는 비워둘 수 없습니다."
            
        if not target_text.strip():
            return None, "생성할 텍스트는 비워둘 수 없습니다."
        
        # 음성 생성 처리
        output_path, was_trimmed, was_resampled = generate_zero_shot_tts(reference_audio, reference_text, target_text)
        
        # 결과 반환
        message_parts = ["생성 완료!"]
        if was_resampled:
            message_parts.append("오디오가 16kHz로 리샘플링되었습니다.")
        if was_trimmed:
            message_parts.append("참조 오디오가 10초로 잘렸습니다.")
            
        message = " ".join(message_parts)
        
        return output_path, message
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"오류: {str(e)}"

# Gradio 인터페이스 생성
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("""
    # 🎤 제로샷 텍스트-음성 변환 생성기
    
    참조 오디오 샘플을 업로드하고, 해당 오디오에서 말한 텍스트를 입력한 다음, 같은 목소리로 생성하고 싶은 새 텍스트를 입력하세요.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 참조 오디오 및 텍스트")
            reference_audio = gr.Audio(label="참조 오디오 업로드 (최대 10초)", type="filepath")
            
            auto_transcribe = gr.Checkbox(
                label="자동 전사 (Whisper Large V3 Turbo 사용)", 
                value=False,
                interactive=WHISPER_AVAILABLE
            )
            
            if not WHISPER_AVAILABLE:
                gr.Markdown("⚠️ 전사 기능이 비활성화되었습니다: Whisper 모델을 로드할 수 없습니다.")
            
            process_btn = gr.Button("오디오 처리", variant="primary")
            audio_status = gr.Textbox(label="상태", interactive=False)
            
            reference_text = gr.Textbox(
                label="참조 텍스트 (오디오에 담긴 내용)", 
                lines=3, 
                placeholder="참조 오디오에서 말하는 텍스트를 입력하세요..."
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 음성 생성")
            target_text = gr.Textbox(
                label="생성할 텍스트", 
                lines=5, 
                placeholder="동일한 목소리로 생성하고 싶은 텍스트를 입력하세요..."
            )
            generate_btn = gr.Button("음성 생성", variant="primary")
            generation_status = gr.Textbox(label="상태", interactive=False)
            output_audio = gr.Audio(label="생성된 오디오")
            
            # 마크다운 텍스트를 output_audio 아래로 이동
            gr.Markdown("""
            ### ℹ️ 참고사항
            - 모든 오디오는 자동으로 16kHz로 리샘플링됩니다
            - 참조 오디오가 10초보다 길면 자동으로 잘립니다
            - 참조 오디오의 품질이 음성 클로닝 품질에 영향을 미칩니다
            - 최상의 결과를 위해 배경 잡음이 적은 깨끗한 오디오를 사용하세요
            - Whisper Large V3 Turbo 모델을 사용하여 자동 전사 기능을 제공합니다
            """)
    
    # 이벤트 핸들러 설정
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

# 앱 실행
if __name__ == "__main__":
    try:
        # 공유 링크 생성을 위해 share=True 강제 적용
        print("공유 링크 생성을 위해 share=True로 Gradio 실행 중...")
        demo.launch(share=True, server_name="0.0.0.0")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Gradio 실행 오류: {e}")
        # 첫 번째 방법이 실패하면 대체 실행 방법 시도
        try:
            print("대체 실행 방법 시도 중...")
            demo.launch(share=True)
        except Exception as e2:
            print(f"대체 실행 방법도 실패: {e2}")
