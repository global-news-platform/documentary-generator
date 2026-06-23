"""
Documentary Pipeline — Free, high-quality long-form video generation.
Runs on Kaggle T4 (16GB VRAM) with zero API costs.

Pipeline:
  1. Script Generation (template or local LLM)
  2. Image Generation (SDXL on T4)
  3. Optional Video Motion (Stable Video Diffusion on T4)
  4. TTS Narration (Edge-TTS, no API key)
  5. Assembly (FFmpeg + MoviePy)
"""

import json
import os
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    topic: str = "The History of Artificial Intelligence"
    output_dir: str = "./output"
    num_scenes: int = 6
    fps: int = 24
    resolution: tuple = (1024, 576)
    image_model: str = "SDXL"  # SDXL or SD3.5
    enable_video_motion: bool = True
    enable_background_music: bool = True
    enable_subtitles: bool = True
    voice: str = "en-US-GuyNeural"
    tts_engine: str = "edge_tts"  # edge_tts or kokoro
    use_local_llm: bool = False
    hf_cache_dir: str = "./hf_cache"


# ──────────────────────────────────────────────
#  Script Generation
# ──────────────────────────────────────────────

DOCUMENTARY_TEMPLATES = {
    "default": {
        "title": None,
        "scenes": [],
    }
}

DEFAULT_SCRIPT_TEMPLATE = """Topic: {topic}

Write a documentary script with {num_scenes} scenes. For each scene provide:
- scene_title: A short title
- narration: 3-5 sentences of voiceover narration
- image_prompt: A detailed visual description for image generation (40-60 words, cinematic style)

Output as a JSON array of objects with keys: scene_title, narration, image_prompt.

SCENE 1:
Title: Introduction to {topic}
Narration: [3-5 sentences introducing the topic]
Image Prompt: [detailed cinematic visual description]

SCENE 2:
Title: [subtopic]
Narration: [3-5 sentences]
Image Prompt: [detailed cinematic visual description]

...continue for all {num_scenes} scenes...
"""


class ScriptGenerator:
    """Generates a documentary script from a topic."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate(self) -> dict:
        """Generate a complete script with scenes."""
        script = {
            "topic": self.config.topic,
            "scenes": self._generate_scenes(),
        }
        return script

    def _generate_scenes(self) -> list:
        """Generate scenes using template-based approach (no API key needed)."""
        topic = self.config.topic
        n = self.config.num_scenes
        scenes = []

        templates = self._get_scene_templates(topic, n)
        for i, t in enumerate(templates):
            scenes.append({
                "scene_id": i + 1,
                "scene_title": t["title"],
                "narration": t["narration"],
                "image_prompt": t["image_prompt"],
                "duration_sec": t.get("duration", 10),
            })

        return scenes

    def _get_scene_templates(self, topic, n):
        """Return pre-written scene templates based on topic."""
        generic_templates = [
            {
                "title": f"Introduction to {topic}",
                "narration": f"Welcome to our documentary exploring {topic}. "
                             f"This fascinating subject has shaped our world in countless ways. "
                             f"Join us as we journey through its origins, evolution, and impact. "
                             f"From humble beginnings to groundbreaking discoveries, this is the story of {topic}.",
                "image_prompt": f"Cinematic wide shot of a dramatic landscape at sunrise, representing the dawn of {topic}, "
                               f"golden hour lighting, professional photography, 8K resolution, National Geographic style",
                "duration": 12,
            },
            {
                "title": f"The Origins of {topic}",
                "narration": f"The roots of {topic} stretch back further than many realize. "
                            f"Early pioneers laid the groundwork for what would become a revolutionary field. "
                            f"Their vision and determination set the stage for transformative change.",
                "image_prompt": f"Historical photograph style showing early laboratory or workshop related to {topic}, "
                               f"vintage aesthetic, warm sepia tones, scientists working with early equipment, documentary style",
                "duration": 10,
            },
            {
                "title": f"Key Breakthroughs in {topic}",
                "narration": f"Several key breakthroughs propelled {topic} forward. "
                            f"Each discovery built upon the last, creating a cascade of innovation. "
                            f"These moments changed everything and opened new frontiers of possibility.",
                "image_prompt": f"Dramatic visualization of a major scientific breakthrough, glowing elements, "
                               f"particles of light, technological advancement, cinematic lighting, epic scale",
                "duration": 10,
            },
            {
                "title": f"How {topic} Affects Our World",
                "narration": f"Today, {topic} touches nearly every aspect of modern life. "
                            f"From healthcare to communication, its influence is everywhere. "
                            f"We see its impact in the tools we use and the world we build.",
                "image_prompt": f"Modern cityscape showing technology and innovation connected to {topic}, "
                               f"futuristic architecture, digital networks visualized as light trails, cyberpunk aesthetic",
                "duration": 10,
            },
            {
                "title": f"Challenges Facing {topic}",
                "narration": f"Despite its progress, {topic} faces significant challenges. "
                            f"Ethical questions, technical limitations, and societal concerns all demand attention. "
                            f"How we address these will shape the future of the field.",
                "image_prompt": f"Contemplative scene showing a person facing a massive digital interface, "
                               f"blue and purple neon lighting, questions and symbols floating, thoughtful mood",
                "duration": 10,
            },
            {
                "title": f"The Future of {topic}",
                "narration": f"Looking ahead, the future of {topic} is filled with possibility. "
                            f"Emerging trends point toward even greater integration into our lives. "
                            f"The journey is far from over — the best chapters may still be unwritten.",
                "image_prompt": f"Utopian future scene representing the bright future of {topic}, "
                               f"floating cities, holographic displays, diverse people working together, optimistic lighting",
                "duration": 12,
            },
        ]

        if n <= len(generic_templates):
            return generic_templates[:n]
        else:
            result = []
            while len(result) < n:
                for t in generic_templates:
                    if len(result) >= n:
                        break
                    copy = dict(t)
                    copy["scene_id"] = len(result) + 1
                    result.append(copy)
            return result


# ──────────────────────────────────────────────
#  Image Generation (SDXL on T4)
# ──────────────────────────────────────────────

class ImageGenerator:
    """Generates high-quality images using SDXL on T4 GPU."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipe = None
        self.device = None

    def load_model(self):
        """Load SDXL pipeline onto GPU."""
        import torch
        from diffusers import DiffusionPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ImageGen] Loading SDXL on {self.device}...")

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.config.hf_cache_dir,
            use_safetensors=True,
        )

        if self.device == "cuda":
            self.pipe = self.pipe.to("cuda")
            self.pipe.enable_model_cpu_offload()

        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        print("[ImageGen] SDXL loaded successfully.")

    def unload_model(self):
        """Free GPU memory."""
        if self.pipe is not None:
            import torch
            del self.pipe
            self.pipe = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("[ImageGen] Model unloaded, GPU memory freed.")

    def generate(self, prompt: str, output_path: str, negative_prompt: str = None) -> str:
        """Generate a single image from a prompt."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if negative_prompt is None:
            negative_prompt = "blurry, low quality, distorted, ugly, deformed, text, watermark"

        print(f"[ImageGen] Generating: {prompt[:60]}...")

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.config.resolution[0],
            height=self.config.resolution[1],
            num_inference_steps=25,
            guidance_scale=7.0,
        ).images

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path, quality=95)
        print(f"[ImageGen] Saved: {output_path}")
        return output_path

    def generate_all(self, scenes: list, image_dir: str) -> list:
        """Generate images for all scenes."""
        self.load_model()
        image_paths = []
        try:
            for scene in scenes:
                prompt = scene["image_prompt"]
                scene_id = scene["scene_id"]
                path = os.path.join(image_dir, f"scene_{scene_id:02d}.png")
                self.generate(prompt, path)
                image_paths.append(path)
        finally:
            self.unload_model()
        return image_paths


# ──────────────────────────────────────────────
#  Video Motion (Stable Video Diffusion)
# ──────────────────────────────────────────────

class VideoMotionGenerator:
    """Generates short video clips from images using SVD on T4."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipe = None
        self.device = None

    def load_model(self):
        """Load Stable Video Diffusion pipeline."""
        import torch
        from diffusers import StableVideoDiffusionPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VideoMotion] Loading SVD on {self.device}...")

        model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.config.hf_cache_dir,
            variant="fp16",
        )

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

        print("[VideoMotion] SVD loaded successfully.")

    def unload_model(self):
        """Free GPU memory."""
        if self.pipe is not None:
            import torch
            del self.pipe
            self.pipe = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("[VideoMotion] SVD unloaded, GPU memory freed.")

    def generate_clip(self, image_path: str, output_path: str, num_frames: int = 25):
        """Generate a short video clip from an image."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from PIL import Image

        print(f"[VideoMotion] Generating clip from: {os.path.basename(image_path)}")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.resolution[0], self.config.resolution[1]))

        frames = self.pipe(
            image,
            decode_chunk_size=8,
            num_frames=num_frames,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
        ).frames[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        import imageio
        writer = imageio.get_writer(output_path, fps=self.config.fps // 2, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        print(f"[VideoMotion] Saved: {output_path}")
        return output_path

    def generate_all(self, image_paths: list, video_dir: str, scenes: list) -> list:
        """Generate video clips for all images."""
        self.load_model()
        video_paths = []
        try:
            for i, img_path in enumerate(image_paths):
                duration = scenes[i].get("duration_sec", 10) if i < len(scenes) else 10
                num_frames = min(int(duration * (self.config.fps // 2)), 25)
                out_path = os.path.join(video_dir, f"clip_{i+1:02d}.mp4")
                self.generate_clip(img_path, out_path, num_frames=num_frames)
                video_paths.append(out_path)
        finally:
            self.unload_model()
        return video_paths


# ──────────────────────────────────────────────
#  TTS Narration (Edge-TTS — free, no API key)
# ──────────────────────────────────────────────

class TTSGenerator:
    """Generates narration audio using Edge-TTS (free, no API key)."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate_scene_audio(self, text: str, output_path: str) -> str:
        """Generate TTS audio for a single scene."""
        import edge_tts

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.config.tts_engine == "edge_tts":
            voice = self.config.voice or "en-US-GuyNeural"
            communicate = edge_tts.Communicate(text, voice)
            communicate.save(output_path)
        else:
            try:
                from kokoro import KPipeline
                pipeline = KPipeline(lang_code="a")
                gen = pipeline(text, voice=self.config.voice or "af_sarah")
                all_audio = []
                for _, _, audio in gen:
                    all_audio.append(audio)
                if all_audio:
                    import torch
                    combined = torch.cat(all_audio, dim=-1)
                    import soundfile as sf
                    sf.write(output_path, combined.numpy(), 24000)
                else:
                    raise RuntimeError("Kokoro generated no audio")
            except ImportError:
                print("[TTS] Kokoro not available, falling back to edge-tts")
                import edge_tts
                communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
                communicate.save(output_path)

        print(f"[TTS] Saved: {output_path}")
        return output_path

    def generate_all(self, scenes: list, audio_dir: str) -> list:
        """Generate narration audio for all scenes."""
        audio_paths = []
        for scene in scenes:
            text = scene["narration"]
            path = os.path.join(audio_dir, f"scene_{scene['scene_id']:02d}.mp3")
            self.generate_scene_audio(text, path)
            audio_paths.append(path)
        return audio_paths


# ──────────────────────────────────────────────
#  Video Assembly (FFmpeg + MoviePy)
# ──────────────────────────────────────────────

class VideoAssembler:
    """Assembles final documentary using FFmpeg."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[WARN] FFmpeg not found. Install FFmpeg for video assembly.")

    def create_scene_video(self, image_path: str, audio_path: str, output_path: str,
                           duration: float = 10.0) -> str:
        """Create a video clip from an image + audio with Ken Burns effect."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res = f"{self.config.resolution[0]}:{self.config.resolution[1]}"

        zoom = 1.02
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={self.config.resolution[0]}:{self.config.resolution[1]}:force_original_aspect_ratio=decrease,pad={self.config.resolution[0]}:{self.config.resolution[1]}:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0005,{zoom})':d={int(duration * self.config.fps)}:s={self.config.resolution[0]}x{self.config.resolution[1]}:fps={self.config.fps}",
            "-c:a", "aac",
            "-shortest",
            "-crf", "18",
            "-preset", "fast",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[Assembly] Scene video: {output_path}")
        return output_path

    def create_title_card(self, title: str, output_path: str, duration: float = 5.0) -> str:
        """Create a title card video using FFmpeg."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        res = f"{self.config.resolution[0]}:{self.config.resolution[1]}"

        filter_complex = (
            f"color=c=#0a0a0a:s={res}:d={duration}:r={self.config.fps},"
            f"drawtext=text='{title}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2-30:fontfile='C\\:/Windows/Fonts/arial.ttf',"
            f"drawtext=text='A Documentary':fontcolor=#AAAAAA:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2+30:fontfile='C\\:/Windows/Fonts/arial.ttf'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", filter_complex,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "18",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[Assembly] Title card: {output_path}")
        return output_path

    def create_outro_card(self, output_path: str, duration: float = 4.0) -> str:
        """Create an outro card."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        res = f"{self.config.resolution[0]}:{self.config.resolution[1]}"

        filter_complex = (
            f"color=c=#0a0a0a:s={res}:d={duration}:r={self.config.fps},"
            f"drawtext=text='Thanks for Watching':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=(h-text_h)/2-20:fontfile='C\\:/Windows/Fonts/arial.ttf',"
            f"drawtext=text='Created with Open Source AI':fontcolor=#888888:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2+30:fontfile='C\\:/Windows/Fonts/arial.ttf'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", filter_complex,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "18",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[Assembly] Outro card: {output_path}")
        return output_path

    def assemble(self, scene_videos: list, audio_paths: list, scenes: list,
                 output_path: str, title: str = None) -> str:
        """Assemble the final documentary video with all scenes."""
        temp_dir = os.path.join(self.config.output_dir, "temp_assembly")
        os.makedirs(temp_dir, exist_ok=True)

        segment_files = []

        if title:
            title_card = self.create_title_card(
                title or self.config.topic,
                os.path.join(temp_dir, "00_title.mp4"),
                duration=5.0,
            )
            segment_files.append(title_card)

        for i, video_path in enumerate(scene_videos):
            segment_files.append(video_path)

        outro = self.create_outro_card(
            os.path.join(temp_dir, "99_outro.mp4"),
            duration=4.0,
        )
        segment_files.append(outro)

        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for seg in segment_files:
                abs_path = Path(seg).resolve().as_posix()
                f.write(f"file '{abs_path}'\n")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            "-movflags", "+faststart",
            output_path,
        ]

        print(f"[Assembly] Rendering final video: {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[Assembly] Done! Video saved to: {output_path}")

        return output_path


# ──────────────────────────────────────────────
#  Main Pipeline Orchestrator
# ──────────────────────────────────────────────

class DocumentaryPipeline:
    """Orchestrates the full documentary generation pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._setup_dirs()

    def _setup_dirs(self):
        base = self.config.output_dir
        for d in ["images", "audio", "videos", "temp_assembly", "final"]:
            os.makedirs(os.path.join(base, d), exist_ok=True)

    def generate_script(self) -> dict:
        """Stage 1: Generate documentary script."""
        print("=" * 60)
        print("STAGE 1/5: Generating Script")
        print("=" * 60)
        gen = ScriptGenerator(self.config)
        script = gen.generate()

        print(f"\nTopic: {script['topic']}")
        for scene in script["scenes"]:
            print(f"  Scene {scene['scene_id']}: {scene['scene_title']}")
        print()

        script_path = os.path.join(self.config.output_dir, "script.json")
        with open(script_path, "w") as f:
            json.dump(script, f, indent=2)

        return script

    def generate_images(self, script: dict) -> list:
        """Stage 2: Generate images for each scene."""
        print("=" * 60)
        print("STAGE 2/5: Generating Images (SDXL on T4)")
        print("=" * 60)

        gen = ImageGenerator(self.config)
        image_dir = os.path.join(self.config.output_dir, "images")
        image_paths = gen.generate_all(script["scenes"], image_dir)
        print(f"Generated {len(image_paths)} images.\n")
        return image_paths

    def generate_video_clips(self, image_paths: list, scenes: list) -> list:
        """Stage 3: Generate video clips from images (optional)."""
        if not self.config.enable_video_motion:
            print("SKIP: Video motion disabled. Using still images with Ken Burns effect.")
            return image_paths

        print("=" * 60)
        print("STAGE 3/5: Generating Video Motion (SVD on T4)")
        print("=" * 60)

        gen = VideoMotionGenerator(self.config)
        video_dir = os.path.join(self.config.output_dir, "videos")

        try:
            video_paths = gen.generate_all(image_paths, video_dir, scenes)
            print(f"Generated {len(video_paths)} video clips.\n")
            return video_paths
        except Exception as e:
            print(f"[WARN] Video motion failed: {e}")
            print("Falling back to still images with Ken Burns effect.")
            return image_paths

    def generate_audio(self, script: dict) -> list:
        """Stage 4: Generate narration audio."""
        print("=" * 60)
        print("STAGE 4/5: Generating Narration (Edge-TTS)")
        print("=" * 60)

        gen = TTSGenerator(self.config)
        audio_dir = os.path.join(self.config.output_dir, "audio")
        audio_paths = gen.generate_all(script["scenes"], audio_dir)
        print(f"Generated {len(audio_paths)} audio clips.\n")
        return audio_paths

    def assemble_video(self, image_or_video_paths: list, audio_paths: list,
                       script: dict) -> str:
        """Stage 5: Assemble final video."""
        print("=" * 60)
        print("STAGE 5/5: Assembling Final Video")
        print("=" * 60)

        assembler = VideoAssembler(self.config)
        scenes = script["scenes"]

        if self.config.enable_video_motion and not str(image_or_video_paths[0]).endswith(".mp4"):
            scene_videos = []
            for i, img_path in enumerate(image_or_video_paths):
                duration = scenes[i].get("duration_sec", 10)
                out_path = os.path.join(self.config.output_dir, "temp_assembly", f"scene_{i+1:02d}.mp4")
                assembler.create_scene_video(img_path, audio_paths[i], out_path, duration)
                scene_videos.append(out_path)
        else:
            scene_videos = image_or_video_paths

        output_path = os.path.join(self.config.output_dir, "final", "documentary.mp4")
        assembler.assemble(
            scene_videos,
            audio_paths,
            scenes,
            output_path,
            title=self.config.topic,
        )
        return output_path

    def run(self) -> str:
        """Run the complete pipeline end-to-end."""
        import time
        start = time.time()

        print()
        print("╔══════════════════════════════════════════════════╗")
        print("║     AI Documentary Generator — Free Pipeline    ║")
        print("╚══════════════════════════════════════════════════╝")
        print(f"Topic: {self.config.topic}")
        print(f"Scenes: {self.config.num_scenes}")
        print(f"Output: {self.config.output_dir}")
        print()

        script = self.generate_script()
        images = self.generate_images(script)
        audio = self.generate_audio(script)

        if self.config.enable_video_motion:
            videos = self.generate_video_clips(images, script["scenes"])
            final = self.assemble_video(videos, audio, script)
        else:
            final = self.assemble_video(images, audio, script)

        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)

        print()
        print("╔══════════════════════════════════════════════════╗")
        print(f"║     COMPLETE in {mins}m {secs}s                      ║")
        print(f"║     Output: {final}")
        print("╚══════════════════════════════════════════════════╝")

        return final


# ──────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Documentary Generator")
    parser.add_argument("--topic", default="The History of Artificial Intelligence",
                        help="Documentary topic")
    parser.add_argument("--scenes", type=int, default=6, help="Number of scenes")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--no-video-motion", action="store_true",
                        help="Disable video motion (use still images)")
    parser.add_argument("--no-music", action="store_true",
                        help="Disable background music")
    parser.add_argument("--resolution", default="1024x576", help="Output resolution")
    parser.add_argument("--voice", default="en-US-GuyNeural", help="TTS voice")

    args = parser.parse_args()
    w, h = args.resolution.split("x")

    config = PipelineConfig(
        topic=args.topic,
        output_dir=args.output,
        num_scenes=args.scenes,
        resolution=(int(w), int(h)),
        enable_video_motion=not args.no_video_motion,
        enable_background_music=not args.no_music,
        voice=args.voice,
    )

    pipeline = DocumentaryPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
