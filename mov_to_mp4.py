from moviepy import VideoFileClip
import argparse
import os
import subprocess
import json

def get_duration_ffprobe(filename):
    try:
        # Use ffprobe to get duration
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            filename
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # Try to get duration from format first
        duration = float(data['format']['duration'])
        print(f"FFprobe detected duration: {duration:.2f} seconds")
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration via ffprobe: {str(e)}")
        return None

def convert_mov_to_mp4(input_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.mp4'
    
    input_video = None
    output_video = None
    try:
        # Get accurate duration first
        ffprobe_duration = get_duration_ffprobe(input_path)
        
        # Load the input video file
        input_video = VideoFileClip(input_path)
        moviepy_duration = input_video.duration
        
        print(f"MoviePy detected duration: {moviepy_duration:.2f} seconds")
        
        # Use ffprobe duration if available, otherwise fallback to moviepy
        original_duration = ffprobe_duration or moviepy_duration
        
        # Write the video file as MP4 with adjusted settings
        input_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=input_video.fps,
            preset='medium',
            ffmpeg_params=[
                "-avoid_negative_ts", "1",
                "-copyts",
                "-fflags", "+genpts",
                "-t", str(original_duration + 0.5)  # Add small buffer
            ]
        )
        
        # Load and check the output video
        output_video = VideoFileClip(output_path)
        converted_duration = output_video.duration
        
        print(f"Successfully converted {input_path} to {output_path}")
        print(f"Original duration: {original_duration:.2f} seconds")
        print(f"Converted duration: {converted_duration:.2f} seconds")
        
        if abs(original_duration - converted_duration) > 0.1:
            print("WARNING: Duration mismatch detected!")
            print(f"Difference: {abs(original_duration - converted_duration):.2f} seconds")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if input_video is not None:
            input_video.close()
        if output_video is not None:
            output_video.close()

def main():
    parser = argparse.ArgumentParser(description='Convert MOV files to MP4')
    parser.add_argument('input', help='Input MOV file path')
    parser.add_argument('-o', '--output', help='Output MP4 file path (optional)')
    
    args = parser.parse_args()
    convert_mov_to_mp4(args.input, args.output)

if __name__ == "__main__":
    main()