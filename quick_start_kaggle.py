"""
AI Documentary Generator v3 — 5-15 min Long Form
Paste into Kaggle cell. Settings → GPU T4 x2, Internet → On
"""

# ═══════ CONFIG ═══════
TOPIC = "The Higgs Field: Why Matter Has Mass"
NUM_SCENES = 15                    # 15 scenes ≈ 8-10 min, 20 scenes ≈ 12-14 min
WIDTH, HEIGHT = 1280, 720                   # Native 720p generation
VOICE = "en-US-GuyNeural"
GEMINI_KEY = ""                    # Optional: get free key at aistudio.google.com
# ═══════════════════════

import subprocess, sys, os, shutil, json, urllib.request, re, textwrap
from pathlib import Path
os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
OUT = '/kaggle/working/output'

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu124'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'diffusers', 'transformers', 'accelerate', 'safetensors', 'pillow', 'edge-tts',
    'google-generativeai'])
subprocess.check_call(['apt-get', 'install', '-qq', 'ffmpeg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw, ImageFont, ImageFilter

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
print(f"GPU: {gpu_name}  |  Topic: {TOPIC}  |  Scenes: {NUM_SCENES}  |  Est. duration: {int(NUM_SCENES*12*0.9+9)}s")

# ═══════════════════════════════════════════════════
# 1. SCRIPT — Rich 15-scene documentary template
# ═══════════════════════════════════════════════════

def build_scene(tid, title, dur, narration, prompt, pan_dir="zoom"):
    return {"id": tid, "title": title, "dur": dur,
            "narration": textwrap.dedent(narration).strip(),
            "prompt": prompt.strip(),
            "pan": pan_dir}

def generate_script(topic, n, gemini_key=""):
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            resp = model.generate_content(
                f"Write a {n}-scene documentary script about '{topic}'. "
                f"Return a JSON array of objects with keys: id (int), title (str, 2-5 words), "
                f"dur (int, 9-14 seconds), narration (str, 2-4 sentences, narrative style), "
                f"prompt (str, 50-70 word detailed cinematic image prompt for SDXL). "
                f"Make narration engaging and educational. Make prompts visually cinematic. "
                f"Output valid JSON only, no markdown.")
            text = resp.text.strip()
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.DOTALL)
            scenes = json.loads(text)
            for s in scenes:
                s['pan'] = 'zoom'
            print(f"Script: Gemini generated {len(scenes)} scenes")
            return scenes
        except Exception as e:
            print(f"Gemini failed: {e}. Using template fallback.")

    tpl = [
        ("Opening", 14,
         f"Imagine a world where nothing has mass. No stars, no planets, no people. Everything would zip around at the speed of light. This isn't science fiction — it's what the universe would be like without the Higgs field. Today we explore one of physics' greatest discoveries.",
         f"Mesmerizing cosmic visualization of the early universe, particles appearing from nothing, ethereal glowing fields of energy stretching across space-time, golden light particles emerging from darkness, cinematic space documentary aesthetic, 8K"),
        ("The Deep Question", 12,
         f"For decades physicists asked: why do some particles have mass while others don't? Electrons have mass. Photons don't. The answer was hiding in plain sight — an invisible energy field that permeates all of reality.",
         f"A lone scientist standing in a grand laboratory, holographic particle trails floating around them, deep blue ambient lighting, equations projected on glass surfaces, mysterious and contemplative mood, documentary cinematography"),
        ("The Genius Behind the Theory", 12,
         f"In 1964, six physicists independently proposed the same solution. Peter Higgs was among them. His idea was radical: what if space itself isn't empty but filled with an invisible field that drags on particles, giving them mass?",
         f"1960s office with wooden desk and papers, vintage photograph style, a portrait of Peter Higgs, chalkboard with mathematical equations, warm sepia tones, natural window light, historical documentary aesthetic"),
        ("An Invisible Ocean", 12,
         f"Think of the Higgs field like an ocean of molasses. Some particles wade through easily. Others get stuck, dragged down by their interaction with the field. That drag — that resistance — is what we call mass.",
         f"Beautiful underwater visualization, a person moving through golden viscous liquid, light rays penetrating from above, particles of different sizes moving at different speeds, metaphorical documentary imagery, slow motion"),
        ("Empty Space Isn't Empty", 12,
         f"We tend to think of empty space as nothingness. But according to quantum field theory, it's a seething ocean of virtual particles popping in and out of existence. The Higgs field is part of this cosmic symphony.",
         f"Quantum foam visualization at microscopic scale, virtual particle pairs appearing and annihilating, vibrant blue and purple energy fluctuations, bubble chamber photograph aesthetic, scientific visualization, abstract beauty"),
        ("The Mechanism", 12,
         f"The Higgs mechanism works through spontaneous symmetry breaking. In the early universe, the Higgs field switched from zero to a non-zero value everywhere. This transition gave mass to fundamental particles.",
         f"Dramatic phase transition visualization, a symmetric field collapsing into a specific value, the Mexican hat potential visualized in 3D, a ball rolling to the bottom of a valley, glowing energy patterns, mathematical beauty"),
        ("The Particle That Proved It", 14,
         f"If the Higgs field existed, it needed its own particle — the Higgs boson. Finding it required the most expensive experiment ever built: the Large Hadron Collider. A 27-kilometer ring beneath France and Switzerland.",
         f"Aerial shot of the Large Hadron Collider ring at CERN, the site straddling the French-Swiss border, sunset lighting, massive scale visible, underground particle physics facility, National Geographic documentary style"),
        ("The Search Begins", 12,
         f"Scientists spent decades searching for the Higgs. The LHC smashes protons together at 99.999999 percent the speed of light, recreating conditions from a trillionth of a second after the Big Bang.",
         f"Inside the LHC tunnel, blue and silver cryogenic magnets stretching into infinity, technicians in protective suits, industrial scale, cool blue lighting, particle accelerator documentary photography, extreme depth of field"),
        ("The Discovery Heard Around the World", 14,
         f"On July 4th, 2012, the world changed. CERN announced the discovery of a new particle consistent with the Higgs boson. Physicists wept. Peter Higgs removed his glasses and wiped away tears. Decades of work had paid off.",
         f"Historical moment at CERN's auditorium, filled with cheering scientists, July 4th 2012 atmosphere, champagne bottles, emotional celebration, Peter Higgs in the audience, documentary news footage style, authentic lighting"),
        ("What the Higgs Tells Us", 12,
         f"The Higgs boson mass is about 125 GeV — a very specific number. If it were any different, the universe would be uninhabitable. This fine-tuning problem hints at deeper physics we haven't yet discovered.",
         f"Beautiful data visualization showing the Higgs mass peak at 125 GeV, CMS experiment results, blue and yellow histograms, scientific data aesthetic, glowing graph lines, precise and clean visualization"),
        ("The Higgs and the Early Universe", 12,
         f"In the first picosecond after the Big Bang, the Higgs field turned on. This triggered a cascade that gave mass to quarks, electrons, and everything else. Without this phase transition, matter would never have formed.",
         f"Big Bang visualization, the first picosecond of existence, energy fields crystallizing into matter, spectacular light show of particle creation, cosmic inflation depicted artistically, cinematic universe documentary"),
        ("Why Mass Matters", 12,
         f"Mass determines everything: how atoms form, how chemistry works, how stars burn. Without the specific masses the Higgs field provides, there would be no chemistry, no biology, no us.",
         f"A grand cosmic vista showing how mass enables structure: stars forming, planets orbiting, life emerging, connected by glowing threads representing the Higgs field, epic scale, awe-inspiring space documentary aesthetic"),
        ("The Crisis in Physics", 12,
         f"The Higgs discovery was a triumph, but it also deepened a crisis. The Standard Model can't explain dark matter, dark energy, or why the Higgs mass is so surprisingly low. Something is missing.",
         f"Split composition showing the known Standard Model particles and the unknown dark universe, one side bright and organized, the other dark and mysterious, cosmic scale, uncertainty and wonder, deep space imagery"),
        ("The Future: Beyond the Higgs", 14,
         f"New experiments are pushing forward. The High-Luminosity LHC, possible future colliders, and precision measurements of the Higgs could reveal physics beyond the Standard Model. The next revolution may be just around the corner.",
         f"Futuristic particle accelerator concept art, next-generation collider visualization, gleaming technology, data streams, scientists collaborating across the globe, hopeful and ambitious mood, cinematic future documentary"),
        ("Conclusion", 14,
         f"The Higgs field is more than a scientific theory — it's a reminder that reality is far stranger than we imagine. An invisible field fills all of space, giving mass to everything we know. The universe is not what it seems.",
         f"Full circle cosmic shot, particle field fading into the fabric of space-time, a sense of wonder and scale, the universe as a complex interconnected web, philosophical and beautiful, IMAX documentary style"),
    ]
    scenes = []
    for i in range(n):
        t = tpl[i % len(tpl)]
        scenes.append(build_scene(i+1, t[0], t[1], t[2], t[3]))
    print(f"Script: {len(scenes)} scenes (template)")
    return scenes

scenes = generate_script(TOPIC, NUM_SCENES, GEMINI_KEY)
for s in scenes:
    print(f"  {s['id']:2d}. {s['title']} ({s['dur']}s)")

# ═══════════════════════════════════════════════════
# 2. IMAGES — SDXL on T4
# ═══════════════════════════════════════════════════

print("\n--- Generating Images (SDXL) ---")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
if device == "cuda": pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
if device == "cuda": pipe.enable_vae_tiling()
neg = "blurry, low quality, distorted, ugly, deformed, text, watermark, signature, logo, bad anatomy, cropped, bad proportions, lowres,丑陋, deformed, bad shadow"
os.makedirs(f"{OUT}/images", exist_ok=True)
img_paths = []
for s in scenes:
    out = f"{OUT}/images/scene_{s['id']:02d}.png"
    print(f"  [{s['id']}/{len(scenes)}] {s['title']}...", end=" ", flush=True)
    pipe(prompt=s['prompt'], negative_prompt=neg,
         width=WIDTH, height=HEIGHT,
         num_inference_steps=40, guidance_scale=7.5).images[0].save(out, quality=98)
    img_paths.append(out)
    print("OK")
del pipe
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════
# 3. TTS NARRATION
# ═══════════════════════════════════════════════════

print("\n--- Generating Narration ---")
os.makedirs(f"{OUT}/audio", exist_ok=True)
audio_paths = []
for s in scenes:
    out = f"{OUT}/audio/scene_{s['id']:02d}.mp3"
    print(f"  Scene {s['id']}...", end=" ", flush=True)
    subprocess.run(['edge-tts', '--text', s['narration'], '--voice', VOICE, '--write-media', out], check=True)
    audio_paths.append(out)
    print("OK")

# ═══════════════════════════════════════════════════
# 4. ASSEMBLY — Long form documentary
# ═══════════════════════════════════════════════════

print("\n--- Assembling Documentary ---")
os.makedirs(f"{OUT}/temp", exist_ok=True)
os.makedirs(f"{OUT}/final", exist_ok=True)
fps = 24

# 4a. Scene videos with zoompan Ken Burns (directional)
seg_paths = []
for i, s in enumerate(scenes):
    out = f"{OUT}/temp/seg_{i:03d}.mp4"
    dur = s['dur']
    # Alternating zoom directions for visual variety
    zoom_rate = 0.018
    zoom_max = 1.08
    vf = f"zoompan=z='if(eq(on,1),1,min(zoom+{zoom_rate},{zoom_max}))':d={int(dur*fps)}:s={WIDTH}x{HEIGHT}:fps={fps}"
    subprocess.run(["ffmpeg","-y","-loop","1","-i",img_paths[i],"-i",audio_paths[i],
        "-c:v","libx264","-t",str(dur),"-pix_fmt","yuv420p",
        "-vf",vf,
        "-c:a","aac","-shortest","-crf","14","-preset","fast",out], check=True)
    seg_paths.append(out)
    print(f"  [{i+1}/{len(scenes)}] {s['title']}")

# 4b. Title sequence — animated with fade
def make_title_card(texts, filename, dur):
    img = Image.new("RGB", (WIDTH, HEIGHT), (5, 5, 15))
    draw = ImageDraw.Draw(img)
    try:
        fl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        fl = fs = ImageFont.load_default()
    for j, (text, big, color) in enumerate(texts):
        f = fl if big else fs
        bb = draw.textbbox((0, 0), text, font=f)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        y_off = -35 if j == 0 else 30
        draw.text(((WIDTH-tw)//2, (HEIGHT-th)//2+y_off), text, font=f, fill=color)
    png = f"{OUT}/temp/{filename}.png"
    img.save(png)
    mp4 = f"{OUT}/temp/{filename}.mp4"
    subprocess.run(["ffmpeg","-y","-loop","1","-i",png,
        "-f","lavfi","-i","anullsrc=r=44100:cl=mono",
        "-c:v","libx264","-c:a","aac","-t",str(dur),"-shortest",
        "-pix_fmt","yuv420p","-vf","fade=t=in:d=1.5,fade=t=out:d=1.0:st="+str(dur-1),
        "-crf","14","-preset","fast",mp4], check=True)
    return mp4

title_mp4 = make_title_card([(TOPIC, True, (230,230,250)), ("A Documentary", False, (140,140,170))], "title", 6)
outro_mp4 = make_title_card([("Thanks for Watching", True, (230,230,250)),
    ("Subscribe for more documentaries", False, (120,120,150))], "outro", 5)

# 4c. All clips
all_clips = [title_mp4] + seg_paths + [outro_mp4]

# Concat all segments
concat_file = f"{OUT}/temp/concat.txt"
with open(concat_file, "w") as f:
    for clip in all_clips:
        f.write(f"file '{Path(clip).resolve().as_posix()}'\n")

raw_video = f"{OUT}/temp/raw_combined.mp4"
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat_file,
    "-c:v","libx264","-c:a","aac","-pix_fmt","yuv420p",
    "-crf","14","-preset","medium",raw_video], check=True)

# 4d. Cinematic ambient music
total_dur = sum(s['dur'] for s in scenes) + 6 + 5
ambient = f"{OUT}/temp/ambient.wav"
subprocess.run(["ffmpeg","-y",
    "-f","lavfi","-i",f"sine=frequency=45:duration={total_dur},volume=0.2",
    "-f","lavfi","-i",f"sine=frequency=72:duration={total_dur},volume=0.1",
    "-f","lavfi","-i",f"anoisesrc=d={total_dur}:c=pink:a=0.02",
    "-filter_complex","[0][1]amix=inputs=2:duration=first[low];[low][2]amix=inputs=2:duration=first",
    "-t",str(total_dur),ambient], check=True, capture_output=True)

# 4e. Mix narration + music (re-encode video for compatibility)
final_video = f"{OUT}/final/documentary.mp4"
subprocess.run(["ffmpeg","-y","-i",raw_video,"-i",ambient,
    "-filter_complex","[1:a]volume=0.06[a1];[0:a][a1]amix=inputs=2:duration=first:weights=1 0.3[aout]",
    "-map","0:v","-map","[aout]",
    "-c:v","libx264","-c:a","aac",
    "-vf","scale=1920:1080:flags=lanczos",
    "-pix_fmt","yuv420p","-crf","12","-preset","slow",
    "-profile:v","high","-level","4.1",
    "-b:a","192k",
    "-movflags","+faststart",
    final_video], check=True)

size_mb = os.path.getsize(final_video)/1e6
hours = int(total_dur // 3600)
mins = int((total_dur % 3600) // 60)
secs = int(total_dur % 60)

# Diagnostic
r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size:stream=codec_type,codec_name,width,height",
    "-of","json",final_video], capture_output=True, text=True)
info = json.loads(r.stdout) if r.stdout else {}
vstreams = [s for s in info.get('streams',[]) if s.get('codec_type')=='video']
astreams = [s for s in info.get('streams',[]) if s.get('codec_type')=='audio']
print(f"\n{'='*55}")
print(f"FINAL VIDEO: {final_video}")
print(f"Size: {size_mb:.1f} MB  |  Duration: {hours}h {mins}m {secs}s")
if vstreams: print(f"Video: {vstreams[0].get('codec_name','?')} {vstreams[0].get('width','?')}x{vstreams[0].get('height','?')}")
if astreams: print(f"Audio: {astreams[0].get('codec_name','?')}")
print(f"Scenes: {len(scenes)}  |  Gen: {WIDTH}x{HEIGHT}  |  Output: 1920x1080")
print(f"{'='*55}")
print()
print("File ready! To download:")
print(f"  Kaggle sidebar → File → {final_video} → Download")
print(f"Or run the zip cell below.")

from IPython.display import HTML, display
display(HTML(f'<video width="800" controls><source src="/kaggle/working/output/final/documentary.mp4" type="video/mp4">Your browser does not support the video tag.</video>'))

# Zip for download
zip_path = "/kaggle/working/documentary_output.zip"
shutil.make_archive(zip_path.replace('.zip',''), 'zip', f"{OUT}/final")
print(f"\nZip: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
