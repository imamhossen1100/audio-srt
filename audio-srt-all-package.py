!pip install -U openai-whisper gradio

import whisper
import gradio as gr

print("Loading Whisper model...")
model = whisper.load_model("medium")

def generate_srt(audio_file, output_language):
    if audio_file is None:
        return "দয়া করে একটি অডিও ফাইল আপলোড করুন!"
    
    # আরবি সিলেক্ট করলে অডিওর অরিজিনাল আরবি ট্রান্সক্রিপশন করবে
    if output_language == "Arabic (আরবি)":
        result = model.transcribe(audio_file, language="ar")
        filename = "arabic_subtitles.srt"
    # ইংরেজি সিলেক্ট করলে সরাসরি ইংরেজিতে অনুবাদ করবে
    else:
        result = model.transcribe(audio_file, task="translate")
        filename = "english_subtitles.srt"
        
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
            
    return filename

app = gr.Interface(
    fn=generate_srt,
    inputs=[
        gr.Audio(type="filepath", label="১. আপনার অডিও ফাইল আপলোড করুন (MP3/WAV)"),
        gr.Radio(["Arabic (আরবি)", "English (ইংরেজি)"], label="২. কোন ভাষার SRT ফাইল চান?", value="Arabic (আরবি)")
    ],
    outputs=gr.File(label="৩. জেনারেট হওয়া SRT ফাইল ডাউনলোড করুন"),
    title="AI Whisper Subtitle Generator",
    description="আরবি বা ইংরেজি সিলেক্ট করে এক ক্লিকে টাইমস্ট্যাম্পসহ SRT ফাইল নামিয়ে নিন।"
)

app.launch(share=True)