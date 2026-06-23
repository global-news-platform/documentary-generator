"""
AI Documentary Generator v2 — High Quality
Paste into Kaggle cell. Settings → GPU T4 x2, Internet → On
"""

TOPIC = "The History of Artificial Intelligence"
NUM_SCENES = 6
WIDTH, HEIGHT = 1024, 576
VOICE = "en-US-GuyNeural"

import subprocess, sys, os, json, shutil
from pathlib import Path
os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
OUT = '/kaggle/working/output'

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu124'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'diffusers', 'transformers', 'accelerate', 'safetensors', 'pillow', 'edge-tts'])
subprocess.check_call(['apt-get', 'install', '-qq', 'ffmpeg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw, ImageFont

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
print(f"GPU: {gpu_name}  |  Topic: {TOPIC}  |  Scenes: {NUM_SCENES}")

# ═══════════════════════════════════
# 1. SCRIPT
# ═══════════════════════════════════

def make_script(topic, n):
    tpl = [
        ("Introduction", 14,
         f"Welcome to {topic}. This documentary explores how {topic.lower()} evolved from theoretical concepts into a force reshaping every aspect of modern civilization, challenging our understanding of intelligence itself.",
         f"Cinematic aerial shot of a futuristic city at golden hour, massive holographic displays showing neural networks, {topic.lower()} visualized as flowing data streams, ultra-detailed, ray tracing, cinematic color grading"),
        ("The Early Foundations", 12,
         f"The roots of {topic} trace back to ancient philosophy and mathematics. But it was the 20th century that saw the first true spark — Alan Turing's vision of machines that could think. From those early dreams, a new field was born.",
         f"Vintage laboratory from the 1950s, early computer mainframes with vacuum tubes, black and white photograph style, dramatic side lighting, documentary aesthetic, grainy film texture"),
        ("The First Breakthroughs", 12,
         f"The 1960s and 70s brought the first wave of breakthroughs. Expert systems emerged, capable of medical diagnosis and chemical analysis. Each success promised a future of intelligent machines. But winter was coming.",
         f"Retro-futuristic 1970s research lab, large tape reel computers, scientists studying printouts, warm amber lighting, control room aesthetic, vintage technology documentary style"),
        ("The Deep Learning Revolution", 12,
         f"Everything changed in 2012 when deep neural networks shattered records in image recognition. Powered by GPUs and massive datasets, machines began to see, hear, and understand. The age of AI had truly arrived.",
         f"Abstract visualization of a deep neural network, colorful interconnected nodes firing with light, GPU servers rack-mounted, data center with blue LED lights, information visualization, cyberpunk aesthetic"),
        ("AI in the Modern World", 12,
         f"Today, {topic.lower()} permeates daily life. From language models that write poetry to systems that diagnose disease faster than doctors, AI has become invisible infrastructure — the electricity of the 21st century.",
         f"Split composition showing AI in everyday life: smartphone, autonomous car, medical scanner, smart home devices, warm ambient lighting, slice-of-life documentary style, ultra-detailed, natural colors"),
        ("The Future and Beyond", 14,
         f"What lies ahead? Artificial general intelligence, human-AI collaboration, perhaps even consciousness. The next decade will determine whether {topic.lower()} becomes humanity's greatest tool or its most profound challenge.",
         f"Mind-bending futuristic visualization of artificial general intelligence, quantum computer cores, holographic interfaces, human and machine collaborating, bioluminescent technology, ethereal lighting, cinematic sci-fi aesthetic"),
    ]
    scenes = []
    for i in range(n):
        t = tpl[i % len(tpl)]
        scenes.append({"id": i+1, "title": t[0], "narration": t[2], "prompt": t[3], "dur": t[1]})
    return scenes

scenes = make_script(TOPIC, NUM_SCENES)
print("Script: OK")

# ═══════════════════════════════════
# 2. IMAGES — SDXL at native 1024x576
# ═══════════════════════════════════

print("\n--- Generating Images (SDXL) ---")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
if device == "cuda": pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
neg = "blurry, low quality, distorted, ugly, deformed, text, watermark, signature, logo, bad anatomy"
os.makedirs(f"{OUT}/images", exist_ok=True)
img_paths = []
for s in scenes:
    out = f"{OUT}/images/scene_{s['id']:02d}.png"
    print(f"  [{s['id']}/{NUM_SCENES}] {s['title']}...", end=" ", flush=True)
    pipe(prompt=s['prompt'], negative_prompt=neg,
         width=WIDTH, height=HEIGHT,
         num_inference_steps=35, guidance_scale=7.5).images[0].save(out, quality=98)
    img_paths.append(out)
    print("OK")
del pipe
torch.cuda.empty_cache()

# ═══════════════════════════════════
# 3. TTS
# ═══════════════════════════════════

print("\n--- Generating Narration ---")
os.makedirs(f"{OUT}/audio", exist_ok=True)
audio_paths = []
for s in scenes:
    out = f"{OUT}/audio/scene_{s['id']:02d}.mp3"
    print(f"  Scene {s['id']}...", end=" ", flush=True)
    subprocess.run(['edge-tts', '--text', s['narration'], '--voice', VOICE, '--write-media', out], check=True)
    audio_paths.append(out)
    print("OK")

# ═══════════════════════════════════
# 4. ASSEMBLY
# ═══════════════════════════════════

print("\n--- Assembling Documentary ---")
os.makedirs(f"{OUT}/temp", exist_ok=True)
os.makedirs(f"{OUT}/final", exist_ok=True)
fps = 24

# 4a. Scene videos with zoompan Ken Burns
seg_paths = []
for i, s in enumerate(scenes):
    out = f"{OUT}/temp/seg_{i:03d}.mp4"
    dur = s['dur']
    subprocess.run(["ffmpeg","-y","-loop","1","-i",img_paths[i],"-i",audio_paths[i],
        "-c:v","libx264","-t",str(dur),"-pix_fmt","yuv420p",
        "-vf",f"zoompan=z='if(eq(on,1),1,min(zoom+0.015,1.06))':d={int(dur*fps)}:s={WIDTH}x{HEIGHT}:fps={fps}",
        "-c:a","aac","-shortest","-crf","16","-preset","fast",out], check=True)
    seg_paths.append(out)
    print(f"  Scene {i+1}: {s['title']} ({dur}s)")

# 4b. Title/outro cards
def make_card(texts, filename, dur):
    img = Image.new("RGB", (WIDTH, HEIGHT), (8, 8, 16))
    draw = ImageDraw.Draw(img)
    try:
        fl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except:
        fl = fs = ImageFont.load_default()
    for j, (text, big, color) in enumerate(texts):
        f = fl if big else fs
        bb = draw.textbbox((0, 0), text, font=f)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        draw.text(((WIDTH-tw)//2, (HEIGHT-th)//2 + (-30 if j==0 else 35)), text, font=f, fill=color)
    p = f"{OUT}/temp/{filename}.png"
    img.save(p)
    m = f"{OUT}/temp/{filename}.mp4"
    subprocess.run(["ffmpeg","-y","-loop","1","-i",p,"-c:v","libx264","-t",str(dur),
        "-pix_fmt","yuv420p","-crf","16","-preset","fast",m], check=True)
    return m

title_mp4 = make_card([(TOPIC, True, (220,220,240)), ("A Documentary", False, (140,140,160))], "title", 5)
outro_mp4 = make_card([("Thanks for Watching", True, (220,220,240)), ("Created with Open Source AI", False, (120,120,140))], "outro", 4)

# 4c. Crossfade between all clips using pairwise xfade
all_clips = [title_mp4] + seg_paths + [outro_mp4]
xfade_dur = 0.8

current = all_clips[0]
for i in range(1, len(all_clips)):
    nxt = all_clips[i]
    out = f"{OUT}/temp/xfade_{i:03d}.mp4"
    # Get duration of current composite
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",current],
        capture_output=True, text=True)
    cur_dur = float(r.stdout.strip())
    offset = cur_dur - xfade_dur
    subprocess.run(["ffmpeg","-y","-i",current,"-i",nxt,
        "-filter_complex",
        f"[0:v][0:a][1:v][1:a]xfade=transition=fade:duration={xfade_dur}:offset={offset}[v];[0:a][1:a]acrossfade=d={xfade_dur}[a]",
        "-map","[v]","-map","[a]",
        "-c:v","libx264","-c:a","aac","-pix_fmt","yuv420p","-crf","16","-preset","medium",out], check=True)
    current = out
    print(f"  Crossfade {i}/{len(all_clips)-1}")

total_dur = sum(s['dur'] for s in scenes) + 5 + 4 - (len(all_clips)-1)*xfade_dur

# 4d. Ambient background music
ambient = f"{OUT}/temp/ambient.wav"
subprocess.run(["ffmpeg","-y","-f","lavfi","-i",
    f"sine=frequency=55:duration={total_dur},volume=0.25",
    "-f","lavfi","-i",
    f"sine=frequency=88:duration={total_dur},volume=0.12",
    "-filter_complex","[0][1]amix=inputs=2:duration=first",
    "-t",str(total_dur),ambient], check=True, capture_output=True)

# 4e. Mix audio (narration + ambient music)
final_video = f"{OUT}/final/documentary.mp4"
subprocess.run(["ffmpeg","-y","-i",current,"-i",ambient,
    "-filter_complex","[1:a]volume=0.06[a1];[0:a][a1]amix=inputs=2:duration=first[aout]",
    "-map","0:v","-map","[aout]","-c:v","copy","-c:a","aac",
    "-movflags","+faststart",final_video], check=True)

size_mb = os.path.getsize(final_video)/1e6
print(f"\n{'='*55}")
print(f"FINAL VIDEO: {final_video}")
print(f"Size: {size_mb:.1f} MB  |  Duration: {total_dur:.0f}s")
print(f"Resolution: {WIDTH}x{HEIGHT}  |  SDXL 35 steps  |  Euler scheduler")
print(f"Ken Burns zoompan + crossfade transitions + ambient music")
print(f"{'='*55}")

from IPython.display import Video
Video(final_video, embed=True, width=800)
