#!/usr/bin/env python3
"""
Video transcription tool using yt-dlp and Whisper.
Downloads audio from video URLs and transcribes using OpenAI's Whisper model.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import whisper
import yt_dlp

# PART I. Download audio from video URL
def download_audio(video_url: str, output_dir: Path = Path(".")) -> Path:
    """Download audio from a video URL and return the file path."""
    print("📥 Downloading audio...")
    
    # Extract video info to get title
    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
        info = ydl.extract_info(video_url, download=False)
    
    # Create safe filename
    title = info.get('title', 'video')
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_title = safe_title[:100]  # Limit length
    
    output_template = str(output_dir / f"{safe_title}.%(ext)s")
    
    # Download options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    audio_file = output_dir / f"{safe_title}.mp3"
    
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not created: {audio_file}")
    
    print(f"✅ Audio downloaded: {audio_file}")
    return audio_file

# PART II. Transcribe audio using Whisper
def transcribe_audio(
    audio_file: Path,
    model_size: str = "turbo",
    language: Optional[str] = None,
    device: Optional[str] = None
) -> str:
    """Transcribe audio file using Whisper."""
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Show GPU info if available
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  Using GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    else:
        print(f"🖥️  Using device: {device}")
    
    print(f"🤖 Loading Whisper model '{model_size}'...")
    
    model = whisper.load_model(model_size, device=device)
    
    print("🎙️  Transcribing...")
    result = model.transcribe(
        str(audio_file),
        language=language,
        fp16=(device == "cuda"),
        verbose=False
    )
    
    # Clear GPU cache if using CUDA
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result["text"].strip()

# PART III. Main function to combine download and transcription
def transcribe_video(
    video_url: str,
    output_file: Optional[Path] = None,
    model_size: str = "turbo",
    language: Optional[str] = None,
    keep_audio: bool = False
) -> str:
    """Download video and transcribe it."""
    audio_file = None
    
    try:
        # Download audio
        audio_file = download_audio(video_url)
        
        # Transcribe
        transcript = transcribe_audio(audio_file, model_size, language)
        
        # Save to file
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(transcript, encoding="utf-8")
            print(f"💾 Transcript saved to: {output_file}")
        
        return transcript
        
    finally:
        # Cleanup audio file
        if audio_file and audio_file.exists() and not keep_audio:
            audio_file.unlink()
            print(f"🗑️  Removed temporary audio: {audio_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and transcribe videos using Whisper"
    )
    parser.add_argument(
        "url",
        help="Video URL to transcribe"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("transcript.txt"),
        help="Output file path (default: transcript.txt)"
    )
    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        default="turbo",
        help="Whisper model size (default: small). For 4GB VRAM: use tiny/base/small/turbo"
    )
    parser.add_argument(
        "-l", "--language",
        help="Language code (e.g., en, es, fr). Auto-detect if not specified."
    )
    parser.add_argument(
        "-k", "--keep-audio",
        action="store_true",
        help="Keep audio file after transcription"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=500,
        help="Number of characters to preview (default: 500)"
    )
    
    args = parser.parse_args()
    
    try:
        transcript = transcribe_video(
            args.url,
            args.output,
            args.model,
            args.language,
            args.keep_audio
        )
        
        print("\n" + "="*60)
        print("📄 TRANSCRIPT PREVIEW")
        print("="*60)
        preview = transcript[:args.preview]
        if len(transcript) > args.preview:
            preview += "..."
        print(preview)
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()