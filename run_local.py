"""
Local test runner for the documentary pipeline.
Tests that the pipeline can run end-to-end without GPU (CPU mode).

Usage:
    python run_local.py --topic "My Topic" --scenes 3
    python run_local.py --no-video-motion  (faster for testing)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import PipelineConfig, DocumentaryPipeline

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test documentary pipeline locally")
    parser.add_argument("--topic", default="Test Documentary Topic")
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--output", default="./output_test")
    parser.add_argument("--resolution", default="768x432")
    parser.add_argument("--no-video-motion", action="store_true")
    parser.add_argument("--voice", default="en-US-GuyNeural")
    args = parser.parse_args()

    w, h = args.resolution.split("x")
    config = PipelineConfig(
        topic=args.topic,
        output_dir=args.output,
        num_scenes=args.scenes,
        resolution=(int(w), int(h)),
        enable_video_motion=not args.no_video_motion,
        enable_background_music=False,
        enable_subtitles=False,
        voice=args.voice,
        hf_cache_dir="./hf_cache_test",
    )

    pipeline = DocumentaryPipeline(config)
    final = pipeline.run()
    print(f"\nDone! Output: {final}")


if __name__ == "__main__":
    main()
