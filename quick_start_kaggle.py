"""
Quick-start script for Kaggle — paste the ENTIRE content of this file
into a single Kaggle notebook cell to run the documentary pipeline.
"""

# ═══════════════════════════════════════════════════════════════
#  AI Documentary Generator — Single-Cell Kaggle Quick Start
# ═══════════════════════════════════════════════════════════════
# 1. Kaggle: Settings → Accelerator → GPU T4 x2, Internet → On
# 2. Create a new notebook, paste this entire file into cell 1
# 3. Edit TOPIC below, then run all
# ═══════════════════════════════════════════════════════════════

# ─── CONFIG ───────────────────────────────────────────────────
TOPIC = "The History of Artificial Intelligence"   # ← CHANGE ME
NUM_SCENES = 6                                      # 4-10 scenes
WIDTH, HEIGHT = 1024, 576                           # 16:9 resolution
ENABLE_MOTION = False     # True = SVD video, False = Ken Burns
VOICE = "en-US-GuyNeural" # Edge-TTS voice
# ─────────────────────────────────────────────────────────────

import subprocess, sys, os, json, asyncio, shutil
from pathlib import Path

os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
OUT = '/kaggle/working/output'

# Install deps
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu124'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'pillow', 'edge-tts', 'imageio[ffmpeg]', 'soundfile'])
subprocess.check_call(['apt-get', 'install', '-qq', 'ffmpeg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import torch
import edge_tts
from diffusers import DiffusionPipeline
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")
print(f"Topic: {TOPIC}  |  Scenes: {NUM_SCENES}  |  Motion: {ENABLE_MOTION}")

# ─── SCRIPT ──────────────────────────────────────────────────
scenes = []
titles = [
    f"Introduction to {TOPIC}",
    f"The Origins of {TOPIC}",
    f"Key Breakthroughs in {TOPIC}",
    f"How {TOPIC} Affects Our World",
    f"Challenges Facing {TOPIC}",
    f"The Future of {TOPIC}",
]
narrations = [
    f"Welcome to our documentary exploring {TOPIC}. This fascinating subject has shaped our world in countless ways. Join us as we journey through its origins, evolution, and impact.",
    f"The roots of {TOPIC} stretch back further than many realize. Early pioneers laid the groundwork for what would become a revolutionary field. Their vision and determination set the stage for transformative change.",
    f"Several key breakthroughs propelled {TOPIC} forward. Each discovery built upon the last, creating a cascade of innovation. These moments changed everything and opened new frontiers.",
    f"Today, {TOPIC} touches nearly every aspect of modern life. From healthcare to communication, its influence is everywhere. We see its impact in the tools we use and the world we build.",
    f"Despite its progress, {TOPIC} faces significant challenges. Ethical questions, technical limitations, and societal concerns all demand attention. How we address these will shape the future.",
    f"Looking ahead, the future of {TOPIC} is filled with possibility. Emerging trends point toward even greater integration into our lives. The journey is far from over — the best chapters may still be unwritten.",
]
prompts = [
    f"Cinematic wide shot of a dramatic landscape at sunrise, representing the dawn of {TOPIC}, golden hour lighting, professional photography, 8K resolution, National Geographic style",
    f"Historical photograph style showing early laboratory or workshop related to {TOPIC}, vintage aesthetic, warm sepia tones, scientists working with early equipment, documentary style",
    f"Dramatic visualization of a major scientific breakthrough, glowing elements, particles of light, technological advancement, cinematic lighting, epic scale",
    f"Modern cityscape showing technology and innovation connected to {TOPIC}, futuristic architecture, digital networks visualized as light trails, cyberpunk aesthetic",
    f"Contemplative scene showing a person facing a massive digital interface, blue and purple neon lighting, questions and symbols floating, thoughtful mood",
    f"Utopian future scene representing the bright future of {TOPIC}, floating cities, holographic displays, diverse people working together, optimistic lighting",
]
for i in range(NUM_SCENES):
    idx = i % len(titles)
    scenes.append({"id": i+1, "title": titles[idx], "narration": narrations[idx], "prompt": prompts[idx], "dur": 12 if idx in [0,5] else 10})

# ─── IMAGES (SDXL on T4) ─────────────────────────────────────
print("\n--- Generating Images (SDXL) ---")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True)
if device == "cuda": pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
neg = "blurry, low quality, distorted, ugly, deformed, text, watermark"
os.makedirs(f"{OUT}/images", exist_ok=True)
img_paths = []
for s in scenes:
    out = f"{OUT}/images/scene_{s['id']:02d}.png"
    print(f"  [{s['id']}/{NUM_SCENES}] Generating...", end=" ")
    pipe(prompt=s["prompt"], negative_prompt=neg, width=WIDTH, height=HEIGHT,
         num_inference_steps=25, guidance_scale=7.0).images[0].save(out, quality=95)
    img_paths.append(out)
    print("OK")
del pipe
if device == "cuda": torch.cuda.empty_cache()
print(f"{len(img_paths)} images done!")

# ─── OPTIONAL: VIDEO MOTION (SVD) ────────────────────────────
vid_paths = img_paths
if ENABLE_MOTION:
    print("\n--- Generating Video Clips (SVD) ---")
    from diffusers import StableVideoDiffusionPipeline
    import imageio
    svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16, variant="fp16")
    svd.enable_model_cpu_offload()
    svd.enable_vae_slicing()
    os.makedirs(f"{OUT}/videos", exist_ok=True)
    vid_paths = []
    for i, ip in enumerate(img_paths):
        nf = min(int(scenes[i]["dur"] * 12), 25)
        out = f"{OUT}/videos/clip_{i+1:02d}.mp4"
        print(f"  [{i+1}/{len(img_paths)}] {nf} frames...", end=" ")
        frames = svd(Image.open(ip).convert("RGB"), decode_chunk_size=8,
                     num_frames=nf, motion_bucket_id=127, noise_aug_strength=0.02).frames[0]
        w = imageio.get_writer(out, fps=12, codec="libx264")
        for f in frames: w.append_data(f)
        w.close()
        vid_paths.append(out)
        print("OK")
    del svd
    torch.cuda.empty_cache()
    print(f"{len(vid_paths)} clips done!")

# ─── TTS NARRATION (Edge-TTS) ────────────────────────────────
print("\n--- Generating Narration ---")
async def gen_tts():
    os.makedirs(f"{OUT}/audio", exist_ok=True)
    paths = []
    for s in scenes:
        out = f"{OUT}/audio/scene_{s['id']:02d}.mp3"
        await edge_tts.Communicate(s["narration"], VOICE).save(out)
        paths.append(out)
    return paths
audio_paths = asyncio.run(gen_tts())
print(f"{len(audio_paths)} audio files done!")

# ─── ASSEMBLY (FFmpeg) ──────────────────────────────────────
print("\n--- Assembling Final Video ---")
os.makedirs(f"{OUT}/temp", exist_ok=True); os.makedirs(f"{OUT}/final", exist_ok=True)
res = f"{WIDTH}:{HEIGHT}"; fps = 24
segs = []

for i in range(NUM_SCENES):
    img = vid_paths[i]; aud = audio_paths[i]; dur = scenes[i]["dur"]
    out = f"{OUT}/temp/seg_{i:03d}.mp4"
    if ENABLE_MOTION:
        subprocess.run(["ffmpeg","-y","-i",img,"-i",aud,"-c:v","copy","-c:a","aac","-shortest",out],
                       check=True, capture_output=True)
    else:
        subprocess.run(["ffmpeg","-y","-loop","1","-i",img,"-i",aud,"-c:v","libx264","-t",str(dur),
            "-pix_fmt","yuv420p",
            "-vf",f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0005,1.02)':d={int(dur*fps)}:s={WIDTH}x{HEIGHT}:fps={fps}",
            "-c:a","aac","-shortest","-crf","18","-preset","fast",out], check=True, capture_output=True)
    segs.append(out)
    print(f"  Segment {i+1}/{NUM_SCENES}")

# Title card
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
subprocess.run(["ffmpeg","-y","-f","lavfi","-i",
    f"color=c=#0a0a0a:s={res}:d=5:r={fps},drawtext=text='{TOPIC}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2-30:fontfile={font},drawtext=text='A Documentary':fontcolor=#AAAAAA:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2+30:fontfile={font}",
    "-c:v","libx264","-pix_fmt","yuv420p","-preset","fast","-crf","18", f"{OUT}/temp/title.mp4"],
    check=True, capture_output=True)
subprocess.run(["ffmpeg","-y","-f","lavfi","-i",
    f"color=c=#0a0a0a:s={res}:d=4:r={fps},drawtext=text='Thanks for Watching':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=(h-text_h)/2-20:fontfile={font},drawtext=text='Created with Open Source AI':fontcolor=#888888:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2+30:fontfile={font}",
    "-c:v","libx264","-pix_fmt","yuv420p","-preset","fast","-crf","18", f"{OUT}/temp/outro.mp4"],
    check=True, capture_output=True)

# Concat
with open(f"{OUT}/temp/concat.txt","w") as f:
    for seg in [f"{OUT}/temp/title.mp4"] + segs + [f"{OUT}/temp/outro.mp4"]:
        f.write(f"file '{Path(seg).resolve().as_posix()}'\\n")

final = f"{OUT}/final/documentary.mp4"
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",f"{OUT}/temp/concat.txt",
    "-c:v","libx264","-c:a","aac","-pix_fmt","yuv420p","-crf","18","-preset","medium","-movflags","+faststart",final],
    check=True, capture_output=True)

size_mb = os.path.getsize(final) / 1e6
print(f"\\n{'='*50}")
print(f"FINAL VIDEO: {final}")
print(f"Size: {size_mb:.1f} MB")
print(f"Duration: {sum(s['dur'] for s in scenes) + 9}s")
print(f"{'='*50}")

from IPython.display import Video
Video(final, embed=True, width=800)
