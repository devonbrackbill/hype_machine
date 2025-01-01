from pynput import mouse
import cv2
import numpy as np
import time
import json
import threading
from datetime import datetime
import argparse
import os
from moviepy import VideoFileClip
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QScreen, QCursor
import sys
import imageio

class ScreenRecorder:
    def __init__(self, output_video="recording.mp4", output_mouse="mouse_positions.json", monitor_number=None):
        self.output_video = output_video
        self.output_mouse = output_mouse
        self.recording = False
        self.mouse_positions = {}
        self.start_time = None
        self.frames = []  # Store frames in memory
        
        # Initialize Qt Application for screen capture
        self.app = QApplication(sys.argv)
        self.screens = QScreen.virtualSiblings(QApplication.primaryScreen())
        
        # Select monitor
        self.list_monitors()
        if monitor_number is None:
            monitor_number = self.select_monitor()
        
        self.selected_screen = self.screens[monitor_number]
        print(f"Selected monitor {monitor_number}: {self.selected_screen.name()}")
        
        # Video settings
        self.fps = 30.0
        
        # Store screen geometry and device pixel ratio for coordinate conversion
        self.screen_geometry = self.selected_screen.geometry()
        self.device_pixel_ratio = self.selected_screen.devicePixelRatio()
        print(f"Screen geometry: {self.screen_geometry.width()}x{self.screen_geometry.height()}")
        print(f"Device pixel ratio: {self.device_pixel_ratio}")

    def list_monitors(self):
        """List all available monitors"""
        print("\nAvailable monitors:")
        for i, screen in enumerate(self.screens):
            geometry = screen.geometry()
            print(f"Monitor {i}: {screen.name()}")
            print(f"  Position: ({geometry.x()}, {geometry.y()})")
            print(f"  Size: {geometry.width()}x{geometry.height()}")
    
    def select_monitor(self):
        """Let user select a monitor"""
        while True:
            try:
                selection = int(input("\nEnter monitor number to record: "))
                if 0 <= selection < len(self.screens):
                    return selection
                else:
                    print("Invalid monitor number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def record_screen(self):
        """Record the screen"""
        preview_scale = 0.5  # Scale down preview
        frame_count = 0
        last_frame_time = time.time()
        target_frame_time = 1.0 / self.fps
        
        while self.recording:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Only capture frame if enough time has elapsed
            if elapsed >= target_frame_time:
                # Capture and process frame
                mouse_pos = QCursor.pos()
                screen_x = int((mouse_pos.x() - self.screen_geometry.x()) * self.device_pixel_ratio)
                screen_y = int((mouse_pos.y() - self.screen_geometry.y()) * self.device_pixel_ratio)
                
                # Store raw screen coordinates with precise timing
                if self.start_time is not None:
                    current_recording_time = time.time() - self.start_time
                    self.mouse_positions[f"{current_recording_time:.3f}"] = [screen_x, screen_y]
                
                # Debug print mouse position occasionally
                if frame_count % 30 == 0:
                    print(f"Screen geometry: {self.screen_geometry.width()}x{self.screen_geometry.height()}")
                    print(f"Mouse raw: ({mouse_pos.x()}, {mouse_pos.y()})")
                    print(f"Mouse screen-relative: ({screen_x}, {screen_y})")
                    print(f"Device pixel ratio: {self.device_pixel_ratio}")
                
                # Capture screen
                pixmap = self.selected_screen.grabWindow(0,
                                                       self.screen_geometry.x(),
                                                       self.screen_geometry.y(),
                                                       self.screen_geometry.width(),
                                                       self.screen_geometry.height())
                
                # Convert to numpy array
                image = pixmap.toImage()
                buffer = image.constBits()
                buffer.setsize(image.sizeInBytes())
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    image.height(), image.width(), 4).copy()
                
                # Draw cursor
                cursor_size = int(20 * self.device_pixel_ratio)  # Scale cursor size
                cursor = np.zeros((cursor_size, cursor_size, 4), dtype=np.uint8)
                # White fill
                cv2.circle(cursor, (cursor_size//2, cursor_size//2), int(6 * self.device_pixel_ratio), (255, 255, 255, 255), -1)
                # Black outline
                cv2.circle(cursor, (cursor_size//2, cursor_size//2), int(6 * self.device_pixel_ratio), (0, 0, 0, 255), 1)
                # Add a black dot in the center
                cv2.circle(cursor, (cursor_size//2, cursor_size//2), int(1 * self.device_pixel_ratio), (0, 0, 0, 255), -1)
                
                cursor_h, cursor_w = cursor.shape[:2]
                x1 = max(0, screen_x - cursor_w//2)
                y1 = max(0, screen_y - cursor_h//2)
                x2 = min(frame.shape[1], x1 + cursor_w)
                y2 = min(frame.shape[0], y1 + cursor_h)
                
                if x2 > x1 and y2 > y1:
                    # Calculate the visible portion of the cursor
                    cursor_x1 = int(0 if x1 >= 0 else -x1)
                    cursor_y1 = int(0 if y1 >= 0 else -y1)
                    cursor_x2 = int(cursor_w - (cursor_w - (x2 - x1)))
                    cursor_y2 = int(cursor_h - (cursor_h - (y2 - y1)))
                    
                    if cursor_x2 > cursor_x1 and cursor_y2 > cursor_y1:
                        # Get the alpha channel for blending
                        alpha = cursor[cursor_y1:cursor_y2, cursor_x1:cursor_x2, 3:] / 255.0
                        # Get the cursor RGB values
                        cursor_rgb = cursor[cursor_y1:cursor_y2, cursor_x1:cursor_x2, :3]
                        
                        # Ensure shapes match
                        frame_region = frame[y1:y2, x1:x2, :3]
                        if frame_region.shape == cursor_rgb.shape:
                            # Blend cursor with frame
                            blended = frame_region * (1 - alpha) + cursor_rgb * alpha
                            frame[y1:y2, x1:x2, :3] = blended
                
                # Store frame with swapped channels for final video
                final_frame = frame[:, :, [2, 1, 0]].copy()
                self.frames.append(final_frame)
                
                # Show preview
                preview = cv2.resize(frame[:, :, :3], None, fx=preview_scale, fy=preview_scale)
                cv2.imshow('Recording Preview (Press "q" to stop)', preview)
                
                # Update timing and count
                frame_count += 1  # Only increment once
                last_frame_time += target_frame_time  # Use fixed time steps
                
                # Debug timing info
                if frame_count % 30 == 0:
                    actual_fps = frame_count / (time.time() - self.start_time)
                    print(f"Current FPS: {actual_fps:.1f}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.recording = False
                    break
            else:
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
        
        return frame_count

    def on_move(self, x, y):
        """Callback for mouse movement"""
        if self.recording and self.start_time is not None:
            current_time = time.time() - self.start_time
            
            # Get screen geometry
            screen_geometry = self.selected_screen.geometry()
            
            # Convert global coordinates to screen-relative coordinates
            screen_x = x - screen_geometry.x()
            screen_y = y - screen_geometry.y()
            
            # Store the screen-relative coordinates
            self.mouse_positions[f"{current_time:.3f}"] = [screen_x, screen_y]
            
            # Debug print to verify coordinates
            if len(self.mouse_positions) % 30 == 0:  # Print every 30th position
                print(f"Mouse pos: Global({x}, {y}) -> Screen({screen_x}, {screen_y})")
                print(f"Screen bounds: {screen_geometry.width()}x{screen_geometry.height()}")
    
    def start_recording(self):
        """Start recording screen and mouse positions"""
        print("Starting recording... Press 'q' in preview window to stop.")
        self.recording = True
        self.start_time = time.time()
        self.frames = []
        
        frame_count = self.record_screen()
        self.stop_recording()
        
        if frame_count == 0:
            raise Exception("No frames were recorded!")
        
        # Calculate actual duration and FPS
        print(f"Recording complete. Saving {frame_count} frames...")
        actual_duration = time.time() - self.start_time
        actual_fps = frame_count / actual_duration
        print(f"Recorded {frame_count} frames in {actual_duration:.1f} seconds ({actual_fps:.1f} fps)")
        
        # Save video using imageio with correct FPS
        try:
            print("Saving video...")
            # Use the actual FPS for saving
            imageio.mimsave(
                self.output_video,
                self.frames,
                fps=actual_fps,  # Use actual recorded FPS
                quality=8,
                macro_block_size=None
            )
            print("Video saved successfully!")
            
            # Save mouse positions
            with open(self.output_mouse, 'w') as f:
                json.dump(self.mouse_positions, f)
            print(f"Mouse positions saved to {self.output_mouse}")
            
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            # Try to save as a different format if MP4 fails
            try:
                backup_file = self.output_video.replace('.mp4', '.gif')
                print(f"Attempting to save as GIF: {backup_file}")
                imageio.mimsave(backup_file, self.frames, fps=self.fps)
                print("Saved as GIF successfully!")
            except Exception as e2:
                print(f"Error saving backup GIF: {str(e2)}")

    def stop_recording(self):
        """Stop recording and save data"""
        self.recording = False
        cv2.destroyAllWindows()
        
        with open(self.output_mouse, 'w') as f:
            json.dump(self.mouse_positions, f)
        
        print(f"Mouse positions saved to {self.output_mouse}")

def create_zoom_effect(frame, t, mouse_positions, zoom_factor=2, zoom_region_size=300):
    """Create zoom effect around mouse position"""
    h, w = frame.shape[:2]
    
    # Find closest timestamp
    current_time = f"{t:.3f}"
    
    # Debug print to see what's happening
    print(f"Time: {current_time}, Available times: {list(mouse_positions.keys())[:5]}...")
    
    # Find the closest timestamp
    closest_time = min(mouse_positions.keys(), 
                      key=lambda x: abs(float(x) - float(current_time)),
                      default=None)
    
    if closest_time is None:
        print(f"No mouse position found for time {current_time}")
        return frame
    
    # Get mouse position
    x, y = mouse_positions[closest_time]
    print(f"Processing frame at time {current_time}, mouse at ({x}, {y})")
    
    # Convert coordinates to integers
    x = int(x)
    y = int(y)
    
    # Check if mouse is off screen
    if x < 0 or y < 0 or x > w or y > h:
        return frame
    
    # Calculate zoom center
    center_x = x
    center_y = y
    
    # Create transformation matrix
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
    
    # Adjust translation to keep mouse position centered
    M[0, 2] += (w/2 - center_x)
    M[1, 2] += (h/2 - center_y)
    
    # Apply affine transformation
    zoomed = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    
    # Create smooth transition mask
    mask = np.zeros((h, w))
    cv2.circle(mask, (w//2, h//2), int(w/3), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (99, 99), 50)
    mask = np.clip(mask, 0, 1)
    mask = mask[:,:,np.newaxis]
    
    # Blend original and zoomed frames
    result = (frame * (1 - mask) + zoomed * mask).astype(np.uint8)
    
    return result

def process_video(input_video, mouse_data, output_video=None, zoom_factor=2):
    """Add zoom effect to video"""
    if output_video is None:
        output_video = os.path.splitext(input_video)[0] + '_zoomed.mp4'
    
    print("Processing video with zoom effect...")
    
    # Load mouse data
    with open(mouse_data, 'r') as f:
        mouse_positions = json.load(f)
    
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise Exception("Could not open input video")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        if not out.isOpened():
            raise Exception("Could not create output video")
        
        # Process frames
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current time
            current_time = frame_number / fps
            
            # Process frame
            processed_frame = create_zoom_effect(
                frame,
                current_time,
                mouse_positions,
                zoom_factor=zoom_factor
            )
            
            # Write frame
            out.write(processed_frame)
            
            # Show progress
            frame_number += 1
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Convert to MP4 using ffmpeg for better compatibility
        temp_output = output_video
        final_output = output_video.replace('.mp4', '_final.mp4')
        
        print("Converting to final MP4 format...")
        import subprocess
        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            final_output
        ], check=True)
        
        # Replace original with converted version
        os.replace(final_output, output_video)
        print("Video processing complete!")
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        return
    
    print(f"Processed video saved to: {output_video}")

def main():
    parser = argparse.ArgumentParser(description='Screen Recorder with Mouse-Following Zoom')
    parser.add_argument('action', choices=['record', 'process'], 
                      help='Action to perform: record new video or process existing video')
    parser.add_argument('--input', help='Input video file (for process action)')
    parser.add_argument('--mouse-data', help='Mouse position data file (for process action)')
    parser.add_argument('--output', help='Output video file (optional)')
    parser.add_argument('--zoom', type=float, default=2.0, 
                      help='Zoom factor for processing (default: 2.0)')
    parser.add_argument('--monitor', type=int, 
                      help='Monitor number to record (will list monitors if not specified)')
    
    args = parser.parse_args()
    
    if args.action == 'record':
        # Generate unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = args.output or f"recording_{timestamp}.mp4"
        mouse_file = f"mouse_{timestamp}.json"
        
        # Record screen and mouse positions
        recorder = ScreenRecorder(video_file, mouse_file, args.monitor)
        recorder.start_recording()
        
        # Automatically process the recording if desired
        if input("\nWould you like to process the recording with zoom effect? (y/n): ").lower() == 'y':
            process_video(video_file, mouse_file, 
                         output_video=os.path.splitext(video_file)[0] + '_zoomed.mp4',
                         zoom_factor=args.zoom)
    
    elif args.action == 'process':
        if not args.input or not args.mouse_data:
            parser.error("process action requires --input and --mouse-data arguments")
        process_video(args.input, args.mouse_data, args.output, args.zoom)

if __name__ == "__main__":
    main()