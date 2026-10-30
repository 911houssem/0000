import gradio as gr
import torch
import tempfile
import os
import requests
import warnings
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from silma_tts.api import SilmaTTS

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. تحميل النماذج (بضغط 4-bit لتوفير الذاكرة)
# ==============================================================================
print("⏳ [1/3] جاري تحميل Jais-ASR (الاستماع)...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="IFM/Jais-ASR",
    trust_remote_code=True,
    device=0 if torch.cuda.is_available() else -1
)

print("⏳ [2/3] جاري تحميل Jais-3-7B (التفكير) بدقة 4-bit...")
model_id = "inception42/Jais-3-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,  # تفعيل الضغط ليعمل على Colab
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("⏳ [3/3] جاري تحميل SILMA TTS (التحدث)...")
tts = SilmaTTS()

# ==============================================================================
# 2. إعداد الصوت المرجعي (سيتم تنزيله تلقائياً إذا لم يكن موجوداً)
# ==============================================================================
REF_AUDIO_PATH = "ar_ref.wav"
REF_AUDIO_TEXT = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."

if not os.path.exists(REF_AUDIO_PATH):
    print("⏳ جاري تحميل ملف الصوت المرجعي...")
    # رابط مباشر لعينة صوتية عربية
    url = "https://huggingface.co/spaces/silma-ai/silma-tts-v1-demo/resolve/main/ar.ref.24k.wav"
    r = requests.get(url, allow_redirects=True)
    with open(REF_AUDIO_PATH, 'wb') as f:
        f.write(r.content)

# ==============================================================================
# 3. خط الأنابيب (Pipeline)
# ==============================================================================
def voice_pipeline(audio_path):
    if not audio_path:
        return "لم يتم إدخال صوت", "الرجاء التحدث أولاً.", None

    try:
        # --- الخطوة 1: الاستماع (ASR) ---
        print("جاري الاستماع...")
        asr_result = asr_pipeline(audio_path, generate_kwargs={"language": "ar"})
        user_text = asr_result["text"].strip()
        if not user_text:
            return "لم يتم التعرف على كلام", "عذراً، لم أسمع شيئاً واضحاً.", None

        # --- الخطوة 2: التفكير (LLM) ---
        print("جاري التفكير...")
        prompt = f"<|user|>\n{user_text}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
        
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.05
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in response_text:
            response_text = response_text.split("<|assistant|>")[-1].strip()
        else:
            response_text = response_text.replace(prompt, "").strip()

        if not response_text:
            response_text = "عذراً، لم أتمكن من صياغة رد."

        # --- الخطوة 3: التحدث (TTS) ---
        print("جاري التحدث...")
        wav, sr, _ = tts.infer(
            ref_file=REF_AUDIO_PATH,
            ref_text=REF_AUDIO_TEXT,
            gen_text=response_text
        )
        
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        import torchaudio
        torchaudio.save(temp_audio.name, wav.unsqueeze(0), sr)
        audio_output_path = temp_audio.name

        return user_text, response_text, audio_output_path

    except Exception as e:
        print(f"❌ خطأ: {e}")
        return "حدث خطأ", f"عذراً، حدث خطأ تقني: {str(e)}", None

# ==============================================================================
# 4. واجهة Gradio
# ==============================================================================
with gr.Blocks(title="المساعد الصوتي العربي", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ المساعد الصوتي العربي المتكامل")
    gr.Markdown("تحدث في الميكروفون، وسيقوم النظام بالاستماع، التفكير، والرد عليك بصوت عربي.")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 تحدث هنا")
            btn = gr.Button("🚀 إرسال", variant="primary", scale=2)
        with gr.Column():
            text_out = gr.Textbox(label="📝 ما سمعته", lines=2)
            reply_out = gr.Textbox(label="🤖 الرد النصي", lines=4)
    
    audio_out = gr.Audio(label="🔊 الرد الصوتي", autoplay=True)
    
    btn.click(
        voice_pipeline,
        inputs=audio_in,
        outputs=[text_out, reply_out, audio_out]
    )

# ==============================================================================
# 5. تشغيل التطبيق
# ==============================================================================
print("✅ كل شيء جاهز! جاري إنشاء الرابط العام...")
demo.launch(share=True, debug=True)