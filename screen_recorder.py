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
from PyQt6.QtWidgets import QApplication, QRubberBand, QWidget
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QScreen, QCursor
import sys
import imageio

class SelectionWindow(QWidget):
    def __init__(self, screen, width=None, height=None):
        super().__init__()
        # Make sure dimensions are even numbers
        if width is not None:
            width = width + (width % 2)  # Make even if odd
        if height is not None:
            height = height + (height % 2)  # Make even if odd
            
        self.screen = screen
        self.rubberband = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.selected_geometry = None
        self.fixed_width = width
        self.fixed_height = height
        
        # Set up full screen transparent window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 50);")
        self.setGeometry(screen.geometry())
        
        # If dimensions are provided, create initial selection in center
        if width and height:
            screen_center = self.geometry().center()
            self.origin = QPoint(
                screen_center.x() - width // 2,
                screen_center.y() - height // 2
            )
            self.rubberband.setGeometry(
                self.origin.x(),
                self.origin.y(),
                width,
                height
            )
            self.rubberband.show()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.fixed_width and self.fixed_height:
                # Move the existing box
                offset_x = self.fixed_width // 2
                offset_y = self.fixed_height // 2
                self.origin = QPoint(
                    event.pos().x() - offset_x,
                    event.pos().y() - offset_y
                )
                self.rubberband.setGeometry(
                    self.origin.x(),
                    self.origin.y(),
                    self.fixed_width,
                    self.fixed_height
                )
            else:
                # Draw new box
                self.origin = event.pos()
                self.rubberband.setGeometry(QRect(self.origin, QPoint()))
            self.rubberband.show()

    def mouseMoveEvent(self, event):
        if self.rubberband.isVisible():
            if self.fixed_width and self.fixed_height:
                # Move the fixed-size box
                screen_rect = self.screen.geometry()
                new_x = max(0, min(event.pos().x() - self.fixed_width // 2, 
                                 screen_rect.width() - self.fixed_width))
                new_y = max(0, min(event.pos().y() - self.fixed_height // 2, 
                                 screen_rect.height() - self.fixed_height))
                self.rubberband.setGeometry(
                    new_x,
                    new_y,
                    self.fixed_width,
                    self.fixed_height
                )
            else:
                # Resize the free-form box
                self.rubberband.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            geometry = self.rubberband.geometry()
            
            # Ensure the region stays within screen bounds
            screen_rect = self.screen.geometry()
            x = max(0, min(geometry.x(), screen_rect.width() - 2))  # Ensure at least 2 pixels width
            y = max(0, min(geometry.y(), screen_rect.height() - 2))  # Ensure at least 2 pixels height
            width = min(screen_rect.width() - x, geometry.width())
            height = min(screen_rect.height() - y, geometry.height())
            
            # Make dimensions even
            width = width + (width % 2)  # Make even if odd
            height = height + (height % 2)  # Make even if odd
            
            # Final bounds check
            width = min(width, screen_rect.width() - x)
            height = min(height, screen_rect.height() - y)
            
            # Ensure minimum dimensions
            width = max(2, width)
            height = max(2, height)
            
            self.selected_geometry = QRect(x, y, width, height)
            print(f"Selected region (after adjustment): {width}x{height} at ({x},{y})")
            self.close()

class ScreenRecorder:
    def __init__(self, output_video="recording.mp4", output_mouse="mouse_positions.json", 
                 monitor_number=None, fixed_width=None, fixed_height=None):
        self.output_video = output_video
        self.output_mouse = output_mouse
        self.recording = False
        self.mouse_positions = {}
        self.click_events = {}
        self.start_time = None
        self.frames = []  # Store frames in memory
        self.mouse_listener = None  # Add this line
        self.recording_region = None  # Will store the selected region
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
        
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
        
        self.scale_factor = self.selected_screen.devicePixelRatio()
        print(f"Screen scale factor: {self.scale_factor}")

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

    def select_recording_region(self):
        """Let user select a region to record or choose full screen"""
        selection_window = SelectionWindow(
            self.selected_screen,
            width=self.fixed_width,
            height=self.fixed_height
        )
        selection_window.show()
        self.app.exec()  # Wait for selection
        
        if selection_window.selected_geometry:
            # Convert to screen coordinates
            region = selection_window.selected_geometry
            self.recording_region = QRect(
                region.x() + self.screen_geometry.x(),
                region.y() + self.screen_geometry.y(),
                region.width(),
                region.height()
            )
            print(f"Selected region: {self.recording_region.width()}x{self.recording_region.height()}")
        else:
            # Use full screen if no selection was made
            self.recording_region = self.screen_geometry
            print("Using full screen recording")

    def get_mouse_position(self):
        """Get current mouse position relative to recording region"""
        cursor = QCursor.pos()
        if self.recording_region:
            # Adjust coordinates relative to the recording region
            relative_x = (cursor.x() - self.recording_region.x()) * self.scale_factor
            relative_y = (cursor.y() - self.recording_region.y()) * self.scale_factor
            
            # Ensure coordinates are within bounds
            relative_x = max(0, min(relative_x, self.recording_region.width() * self.scale_factor))
            relative_y = max(0, min(relative_y, self.recording_region.height() * self.scale_factor))
            
            return relative_x, relative_y
        return cursor.x() * self.scale_factor, cursor.y() * self.scale_factor

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
                if self.recording_region:
                    # Calculate cursor position relative to recording region
                    cursor_x = int((mouse_pos.x() - self.recording_region.x()) * self.scale_factor)
                    cursor_y = int((mouse_pos.y() - self.recording_region.y()) * self.scale_factor)
                    
                    # Ensure cursor stays within bounds
                    cursor_x = max(0, min(cursor_x, self.recording_region.width() * self.scale_factor))
                    cursor_y = max(0, min(cursor_y, self.recording_region.height() * self.scale_factor))
                else:
                    cursor_x = int((mouse_pos.x() - self.screen_geometry.x()) * self.scale_factor)
                    cursor_y = int((mouse_pos.y() - self.screen_geometry.y()) * self.scale_factor)
                
                # Store raw screen coordinates with precise timing
                if self.start_time is not None:
                    current_recording_time = time.time() - self.start_time
                    self.mouse_positions[f"{current_recording_time:.3f}"] = [cursor_x, cursor_y]
                    
                    # Store click events separately and debug print
                    modifiers = QApplication.mouseButtons()
                    if modifiers & Qt.MouseButton.LeftButton:
                        self.click_events[f"{current_recording_time:.3f}"] = [cursor_x, cursor_y]
                        print(f"Click detected at ({cursor_x}, {cursor_y})")
                
                # Debug print mouse position occasionally
                if frame_count % 30 == 0:
                    print(f"Screen geometry: {self.screen_geometry.width()}x{self.screen_geometry.height()}")
                    print(f"Mouse raw: ({mouse_pos.x()}, {mouse_pos.y()})")
                    print(f"Mouse screen-relative: ({cursor_x}, {cursor_y})")
                    print(f"Device pixel ratio: {self.device_pixel_ratio}")
                
                # Modify the screen capture to use the selected region
                if self.recording_region:
                    scaled_width = int(self.recording_region.width() * self.scale_factor)
                    scaled_height = int(self.recording_region.height() * self.scale_factor)
                    pixmap = self.selected_screen.grabWindow(
                        0,
                        self.recording_region.x(),
                        self.recording_region.y(),
                        self.recording_region.width(),
                        self.recording_region.height()
                    )
                    # Scale the captured image to account for Retina display
                    if self.scale_factor != 1:
                        pixmap = pixmap.scaled(scaled_width, scaled_height)
                else:
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
                
                # Draw cursor with adjusted position
                cursor_size = int(20 * self.scale_factor)
                cursor = np.zeros((cursor_size, cursor_size, 4), dtype=np.uint8)
                cv2.circle(cursor, (cursor_size//2, cursor_size//2), int(6 * self.scale_factor), (255, 255, 255, 255), -1)
                cv2.circle(cursor, (cursor_size//2, cursor_size//2), int(6 * self.scale_factor), (0, 0, 0, 255), 1)
                cv2.circle(cursor, (cursor_size//2, cursor_size//2), int(1 * self.scale_factor), (0, 0, 0, 255), -1)
                
                cursor_h, cursor_w = cursor.shape[:2]
                
                # Ensure cursor coordinates are within frame bounds
                cursor_x = int(max(cursor_w//2, min(frame.shape[1] - cursor_w//2, cursor_x)))
                cursor_y = int(max(cursor_h//2, min(frame.shape[0] - cursor_h//2, cursor_y)))
                
                x1 = int(max(0, cursor_x - cursor_w//2))
                y1 = int(max(0, cursor_y - cursor_h//2))
                x2 = int(min(frame.shape[1], x1 + cursor_w))
                y2 = int(min(frame.shape[0], y1 + cursor_h))
                
                # Additional bounds check
                if (x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0] and 
                    x2 > x1 and y2 > y1):
                    # Calculate the visible portion of the cursor
                    cursor_x1 = int(0 if x1 >= 0 else -x1)
                    cursor_y1 = int(0 if y1 >= 0 else -y1)
                    cursor_x2 = int(cursor_w - (cursor_w - (x2 - x1)))
                    cursor_y2 = int(cursor_h - (cursor_h - (y2 - y1)))
                    
                    if (cursor_x2 > cursor_x1 and cursor_y2 > cursor_y1 and 
                        cursor_y2 <= cursor.shape[0] and cursor_x2 <= cursor.shape[1]):
                        # Get the alpha channel for blending
                        alpha = cursor[cursor_y1:cursor_y2, cursor_x1:cursor_x2, 3:] / 255.0
                        # Get the cursor RGB values
                        cursor_rgb = cursor[cursor_y1:cursor_y2, cursor_x1:cursor_x2, :3]
                        
                        # Ensure shapes match before blending
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
    
    def on_click(self, x, y, button, pressed):
        """Callback for mouse clicks"""
        if self.recording and self.start_time is not None and pressed and button == mouse.Button.left:
            # Convert global coordinates to screen-relative coordinates
            screen_x = int((x - self.screen_geometry.x()) * self.device_pixel_ratio)
            screen_y = int((y - self.screen_geometry.y()) * self.device_pixel_ratio)
            
            # Check if click is within screen bounds
            if (0 <= screen_x < self.screen_geometry.width() * self.device_pixel_ratio and 
                0 <= screen_y < self.screen_geometry.height() * self.device_pixel_ratio):
                
                current_time = time.time() - self.start_time
                self.click_events[f"{current_time:.3f}"] = [screen_x, screen_y]
                print(f"Click detected at ({screen_x}, {screen_y})")
            else:
                print(f"Ignoring off-screen click at ({screen_x}, {screen_y})")

    def start_recording(self):
        """Start recording screen and mouse positions"""
        # Add region selection before starting recording
        self.select_recording_region()
        
        print("Starting recording... Press 'q' in preview window to stop.")
        self.recording = True
        self.start_time = time.time()
        self.frames = []
        self.mouse_positions = {}
        self.click_events = {}
        
        # Start mouse listener
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()
        
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
                fps=actual_fps,
                quality=8,
                macro_block_size=None
            )
            print("Video saved successfully!")
            
            # Save mouse positions and click events separately
            data = {
                'positions': self.mouse_positions,
                'clicks': self.click_events
            }
            with open(self.output_mouse, 'w') as f:
                json.dump(data, f)
            print(f"Mouse data saved to {self.output_mouse}")
            
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
        if self.mouse_listener:
            self.mouse_listener.stop()
        cv2.destroyAllWindows()
        
        # Save both mouse positions and click events
        data = {
            'positions': self.mouse_positions,
            'clicks': self.click_events
        }
        
        with open(self.output_mouse, 'w') as f:
            json.dump(data, f)
        
        print(f"Mouse positions and click events saved to {self.output_mouse}")

def create_zoom_effect(frame, current_time, mouse_positions, click_events, zoom_factor=2.0, 
                      zoom_window=1.0, pre_click_ratio=0.15):
    """Create zoom effect based on click events"""
    h, w = frame.shape[:2]
    
    # Initialize static variables
    if not hasattr(create_zoom_effect, 'current_zoom'):
        create_zoom_effect.current_zoom = 1.0
        create_zoom_effect.zoom_center = None
        create_zoom_effect.last_click_time = None
    
    # Find nearest click event
    click_times = [float(t) for t in click_events.keys()]
    if not click_times:
        return frame
    
    # Find the closest click event that's not too far in the future
    valid_clicks = [t for t in click_times if (current_time - t) > -0.1]  # Small lookahead
    if not valid_clicks:
        return frame
    closest_click = max(valid_clicks)  # Use the most recent valid click
    time_to_click = current_time - closest_click
    
    # Get click position
    x, y = click_events[f"{closest_click:.3f}"]
    
    # Check if new click is within current zoom region (with larger tolerance)
    if create_zoom_effect.zoom_center and create_zoom_effect.current_zoom > 1.1:
        old_x, old_y = create_zoom_effect.zoom_center
        zoom_radius_x = (w / create_zoom_effect.current_zoom) * 0.75  # Increased radius
        zoom_radius_y = (h / create_zoom_effect.current_zoom) * 0.75  # Increased radius
        within_zoom = (abs(x - old_x) < zoom_radius_x and 
                      abs(y - old_y) < zoom_radius_y)
    else:
        within_zoom = False
    
    # Calculate zoom based on proximity to click
    adjusted_time = time_to_click + (zoom_window * pre_click_ratio)
    
    if 0 <= adjusted_time <= zoom_window * 2.0:
        if within_zoom:
            # Just pan to new location, maintaining zoom level
            pan_smoothing = 0.85
            new_x = int(pan_smoothing * old_x + (1 - pan_smoothing) * x)
            new_y = int(pan_smoothing * old_y + (1 - pan_smoothing) * y)
            create_zoom_effect.zoom_center = (new_x, new_y)
        else:
            # Regular zoom behavior
            if adjusted_time < zoom_window * 0.3:
                zoom_progress = 1.0 - (adjusted_time / (zoom_window * 0.3))
            elif adjusted_time < zoom_window * 1.7:
                zoom_progress = 1.0
            else:
                zoom_progress = (zoom_window * 2.0 - adjusted_time) / (zoom_window * 0.3)
            
            target_zoom = 1.0 + (zoom_factor - 1.0) * max(0, min(1, zoom_progress))
            zoom_smoothing = 0.85 if zoom_progress > create_zoom_effect.current_zoom else 0.90
            create_zoom_effect.current_zoom = (zoom_smoothing * create_zoom_effect.current_zoom + 
                                             (1 - zoom_smoothing) * target_zoom)
            create_zoom_effect.zoom_center = (int(x), int(y))
        
        create_zoom_effect.last_click_time = closest_click
    else:
        # Quick return to no zoom
        create_zoom_effect.current_zoom = max(1.0, create_zoom_effect.current_zoom * 0.90)
        if abs(create_zoom_effect.current_zoom - 1.0) < 0.01:
            create_zoom_effect.zoom_center = None
    
    # Apply zoom if needed
    if create_zoom_effect.current_zoom > 1.01 and create_zoom_effect.zoom_center:
        center_x, center_y = create_zoom_effect.zoom_center
        window_w = w / create_zoom_effect.current_zoom
        window_h = h / create_zoom_effect.current_zoom
        
        x1 = int(center_x - window_w/2)
        y1 = int(center_y - window_h/2)
        x2 = int(center_x + window_w/2)
        y2 = int(center_y + window_h/2)
        
        # Ensure zoom window stays within frame bounds
        x1 = max(0, min(x1, w - int(window_w)))
        y1 = max(0, min(y1, h - int(window_h)))
        x2 = min(w, x1 + int(window_w))
        y2 = min(h, y1 + int(window_h))
        
        # Extract and resize the region
        zoomed_region = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(zoomed_region, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed
    
    return frame

def process_video(input_video, mouse_data, output_video=None, zoom_factor=2.0,
                 zoom_window=1.0):  # Time window around click (seconds)
    """Add zoom effect to video"""
    if output_video is None:
        output_video = os.path.splitext(input_video)[0] + '_zoomed.mp4'
    
    print("Processing video with zoom effect...")
    
    # Load mouse data
    with open(mouse_data, 'r') as f:
        data = json.load(f)
        mouse_positions = data.get('positions', {})
        click_events = data.get('clicks', {})
    
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
                click_events,
                zoom_factor=zoom_factor,
                zoom_window=zoom_window
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
    parser.add_argument('action', choices=['record', 'process', 'reprocess'], 
                      help='Action to perform: record new video, process existing video, or reprocess existing zoomed video')
    parser.add_argument('--input', help='Input video file (for process/reprocess action)')
    parser.add_argument('--mouse-data', help='Mouse position data file (for process/reprocess action)')
    parser.add_argument('--output', help='Output video file (optional)')
    parser.add_argument('--zoom', type=float, default=4.0, 
                      help='Zoom factor for processing (default: 4.0)')
    parser.add_argument('--monitor', type=int, 
                      help='Monitor number to record (will list monitors if not specified)')
    parser.add_argument('--width', type=int, help='Fixed width of recording region in pixels')
    parser.add_argument('--height', type=int, help='Fixed height of recording region in pixels')
    parser.add_argument('--fullscreen', action='store_true', 
                      help='Record full screen without region selection')
    
    args = parser.parse_args()
    
    if args.action == 'record':
        # Generate unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = args.output or f"recording_{timestamp}.mp4"
        mouse_file = f"mouse_{timestamp}.json"
        
        # Record screen and mouse positions
        recorder = ScreenRecorder(
            video_file, 
            mouse_file, 
            args.monitor,
            fixed_width=args.width,
            fixed_height=args.height
        )
        if not args.fullscreen:
            recorder.select_recording_region()
        recorder.start_recording()
        
        # Automatically process the recording if desired
        if input("\nWould you like to process the recording with zoom effect? (y/n): ").lower() == 'y':
            process_video(video_file, mouse_file, 
                         output_video=os.path.splitext(video_file)[0] + '_zoomed.mp4',
                         zoom_factor=4.0,
                         zoom_window=1.0)  # 1 second window around each click
    
    elif args.action == 'process':
        if not args.input or not args.mouse_data:
            parser.error("process action requires --input and --mouse-data arguments")
        process_video(args.input, args.mouse_data, args.output, args.zoom)
    
    elif args.action == 'reprocess':
        if not args.input:
            parser.error("reprocess action requires --input argument")
        
        # More robust input filename handling
        input_path = args.input
        # Remove .mp4 extension if present
        input_base = input_path[:-4] if input_path.endswith('.mp4') else input_path
        
        # Try different possible original video filenames
        possible_originals = [
            input_base + '.mp4',  # If input was without extension
            input_base.replace('_zoomed', '') + '.mp4',  # If input was zoomed version
            input_base + '_zoomed.mp4'  # If input was original version
        ]
        
        # Find the original video
        original_video = None
        for possible in possible_originals:
            if os.path.exists(possible):
                original_video = possible
                break
        
        if not original_video:
            parser.error(f"Could not find original video file. Tried: {', '.join(possible_originals)}")
        
        # Similarly handle mouse data file
        base_for_mouse = original_video[:-4].replace('recording_', 'mouse_')
        mouse_data = args.mouse_data or f"{base_for_mouse}.json"
        
        if not os.path.exists(mouse_data):
            parser.error(f"Mouse data file not found: {mouse_data}")
        
        # Set output path
        if args.output:
            output = args.output
        else:
            # If input was original, append _zoomed, if it was zoomed, use same name
            output = (input_base + '_zoomed.mp4' if '_zoomed' not in input_base 
                     else input_base + '.mp4')
        
        print(f"Using original video: {original_video}")
        print(f"Using mouse data: {mouse_data}")
        print(f"Output will be saved to: {output}")
        
        # Process the video with new zoom settings
        process_video(original_video, mouse_data, output, args.zoom)

if __name__ == "__main__":
    main()