# T4 GPU enable thaka obosthay ei cell run korun
!pip install -U openai-whisper gradio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import whisper
import gradio as gr
import torch
import os
import base64

# GPU available kina check kore speed optimize kora
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device.upper()} 🚀 (Super Fast Mode)")

print("Loading Whisper Model (Base/Tiny for Max Speed)...")
model = whisper.load_model("base", device=device)
print("Engine Ready!")

def generate_srt(audio_file, output_language):
    if audio_file is None:
        raise gr.Error("⚠️ Doya kore prothome ekti audio file upload korun!")
    
    fp16_val = True if device == "cuda" else False
    
    if output_language == "Arabic (عربي)":
        result = model.transcribe(audio_file, language="ar", fp16=fp16_val)
        filename = "smart_bangla_arabic_subtitles.srt"
    else:
        result = model.transcribe(audio_file, task="translate", fp16=fp16_val)
        filename = "smart_bangla_english_subtitles.srt"
        
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    with open(filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            f.write(f"{i+1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")
            
    # Base64 encoding to make a 100% clickable direct download link in HTML
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    download_html = f"""
    <div style="text-align: center; background: #1f2937; padding: 25px; border-radius: 14px; border: 1px solid #374151;">
        <h3 style="color: #10b981; margin-bottom: 15px;">✅ SRT File Generated Successfully!</h3>
        <p style="color: #9ca3af; margin-bottom: 20px;">Nicher botame click kore direct download kore nin:</p>
        <a href="data:text/plain;base64,{b64}" download="{filename}" 
           style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 14px 28px; border-radius: 10px; text-decoration: none; font-weight: 700; font-size: 1.1rem; display: inline-block; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);">
           📥 Download {filename} Now
        </a>
    </div>
    """
    return download_html

custom_css = """
body { background-color: #0b0f19 !important; font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
.gradio-container { max-width: 100% !important; width: 100% !important; min-height: 100vh; border-radius: 0px !important; box-shadow: none !important; border: none !important; background: #0b0f19 !important; padding: 40px 20px !important; }
.main-box { max-width: 1100px; margin: 0 auto; background: #111827; border: 1px solid #1f2937; border-radius: 20px; padding: 40px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7); }
.brand-badge { background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); color: white; padding: 6px 16px; border-radius: 30px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; display: inline-block; margin-bottom: 15px; }
h1 { color: #f8fafc; font-weight: 800; text-align: center; font-size: 2.6rem; margin-bottom: 10px; }
p { color: #94a3b8; text-align: center; font-size: 1.1rem; }
.yt-btn { background: #dc2626; color: white !important; text-decoration: none; padding: 10px 22px; border-radius: 8px; font-weight: 600; font-size: 0.95rem; display: inline-flex; align-items: center; gap: 8px; transition: background 0.3s ease; }
.yt-btn:hover { background: #b91c1c; }
.primary-btn { background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important; border: none !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; padding: 14px !important; font-size: 1.1rem !important; width: 100%; }
.primary-btn:hover { background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important; box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5); }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="red")) as app:
    
    with gr.Column(elem_classes="main-box"):
        gr.HTML("""
            <div style="text-align: center; padding-bottom: 30px;">
                <div class="brand-badge">Official Tool by Smart Bangla (Turbo Mode)</div>
                <h1>Smart Bangla Audio to Subtitle Studio</h1>
                <p style="margin-bottom: 20px;">Super-fast AI audio processing & SRT generation for Smart Bangla</p>
                <a href="https://www.youtube.com/@SmartBangla-p9r" target="_blank" class="yt-btn">
                    ▶ Subscribe Smart Bangla on YouTube
                </a>
            </div>
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                audio_input = gr.Audio(type="filepath", label="📁 Audio File Select/Upload (MP3 / WAV)")
                lang_radio = gr.Radio(
                    ["Arabic (عربي)", "English (Translate)"], 
                    label="🌐 Subtitle Language & Mode", 
                    value="Arabic (عربي)"
                )
                submit_btn = gr.Button("🚀 Generate Super Fast SRT", variant="primary", elem_classes="primary-btn")
                
            with gr.Column(scale=1):
                # HTML output component jekhane direct clickable download button dekhabe
                file_output = gr.HTML(label="📥 Download Ready SRT")
                
        submit_btn.click(
            fn=generate_srt, 
            inputs=[audio_input, lang_radio], 
            outputs=file_output
        )

        gr.HTML("""
            <div style="text-align: center; margin-top: 40px; color: #64748b; font-size: 0.9rem; border-top: 1px solid #1f2937; padding-top: 20px;">
                Powered by <b>OpenAI Whisper Turbo</b> &bull; Managed & Produced by <a href="https://www.youtube.com/@SmartBangla-p9r" target="_blank" style="color: #ef4444; text-decoration: none; font-weight: 600;">Smart Bangla</a>
            </div>
        """)

app.launch(share=True)
