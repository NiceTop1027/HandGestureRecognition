"""
Hand Gesture Recognition - Premium UI
Minimalist, high-performance hand tracking application

Controls:
- ESC or Q: Quit application
- H: Toggle info panel
- L: Toggle hand landmarks
- G: Toggle gesture guide
"""
import cv2
import mediapipe as mp
import numpy as np
import time
from gesture_detector import GestureDetector
from animation_3d import Animation3D
from particle_system import ParticleSystem


class PremiumUI:
    """Premium UI design with glassmorphism and smooth animations"""
    
    # Color palette - Professional & Premium
    BG_COLOR = (15, 15, 20)           # Dark background
    ACCENT_PRIMARY = (147, 112, 219)  # Purple
    ACCENT_SECONDARY = (100, 200, 255) # Cyan
    TEXT_PRIMARY = (255, 255, 255)    # White
    TEXT_SECONDARY = (180, 180, 190)  # Light gray
    SUCCESS_COLOR = (80, 220, 120)    # Green
    
    def __init__(self):
        self.animation_time = 0
        
    def draw_glassmorphism_panel(self, frame, x, y, w, h, alpha=0.2):
        """
        Draw glassmorphism panel with ROI optimization
        (Faster than full frame blending)
        """
        # Ensure ROI is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, w_frame - x); h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0: return

        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        
        # Create colored overlay for just the ROI
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 55), -1)
        
        # Blend only the ROI
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        
        # Put back blended ROI
        frame[y:y+h, x:x+w] = roi
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 120), 2)
    
    def draw_gradient_text(self, frame, text, x, y, font_scale=1.0, thickness=2):
        """Draw text with gradient effect"""
        # Shadow first
        cv2.putText(frame, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
        
        # Main text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, self.TEXT_PRIMARY, thickness, cv2.LINE_AA)
    
    def draw_smooth_circle(self, frame, center, radius, color, thickness=-1):
        """Draw anti-aliased circle"""
        cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
    
    def draw_hand_hud(self, frame, center, text, color):
        """Draw futuristic HUD around hand (Optimized)"""
        x, y = center
        
        # Rotating rings animation
        self.animation_time += 0.2 # Faster animation
        radius = 50 + int(np.sin(self.animation_time) * 5)
        
        # Outer ring (Simple circle is fast)
        cv2.circle(frame, (x, y), radius, color, 1, cv2.LINE_AA)
        
        # Inner segmented ring - Optimize loop
        # Reduce segments from 8 to 4 or 6 for speed, but keep look
        # Pre-calculate sin/cos if possible, or reduce calls
        offset = self.animation_time * 20
        for i in range(0, 360, 60): # 45 -> 60 (Fewer dots)
            angle = np.radians(i + offset)
            # Fast int conversion
            end_x = int(x + np.cos(angle) * (radius - 10))
            end_y = int(y + np.sin(angle) * (radius - 10))
            cv2.circle(frame, (end_x, end_y), 3, color, -1) # Slightly larger dots, fewer count
            
        # Label
        cv2.putText(frame, text, (x - 30, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Connection line to center (Energy beam)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Draw line only if interacting
        cv2.line(frame, (x, y), (cx, cy), color, 1, cv2.LINE_AA)
        
        # Draw single data point optimization
        # Avoid complex mix math if possible
        mix = (self.animation_time % 5) / 5.0
        dot_x = int(x + (cx - x) * mix) # Optimized lerp
        dot_y = int(y + (cy - y) * mix)
        cv2.circle(frame, (dot_x, dot_y), 4, (255, 255, 255), -1)
    
    def draw_progress_arc(self, frame, center, radius, progress, color):
        """Draw circular progress indicator"""
        # Background arc
        cv2.ellipse(frame, center, (radius, radius), 0, 0, 360,
                   (60, 60, 70), 3, cv2.LINE_AA)
        
        # Progress arc
        angle = int(360 * progress)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle,
                   color, 4, cv2.LINE_AA)


class HandGestureApp:
    """Main application with premium UI"""
    
    TARGET_FPS = 60
    
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Enable Dual Hands
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Components
        self.gesture_detector = GestureDetector()
        self.ui = PremiumUI()
        self.animation_3d = None  # Will be initialized after camera
        self.particle_system = ParticleSystem(max_particles=300)
        
        # UI state
        self.show_info = True
        self.show_landmarks = True
        self.show_guide = False
        self.show_3d = True  # Show 3D animation
        
        # Performance tracking
        self.fps = 0
        self.frame_times = []
        self.last_time = time.time()
        self.prev_time = time.time()
        
        # Gesture control state
        self.last_gesture_type = None
        self.prev_palm_center = None
        self.hand_velocity = np.array([0.0, 0.0])
        
        # Camera
        self.cap = None
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(camera_index)
        
        # HD resolution for quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot access camera")
        
        # Get frame dimensions for 3D animation
        ret, frame = self.cap.read()
        if ret:
            height, width = frame.shape[:2]
            self.animation_3d = Animation3D(width, height)
        
        print("✅ Camera initialized at 1280x720@60fps")
        print("✨ 3D Animation system loaded")
        return True
    
    def update_fps(self):
        """Calculate smooth FPS"""
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        
        if delta > 0:
            self.frame_times.append(1.0 / delta)
            
        # Keep last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate average
        self.fps = int(sum(self.frame_times) / len(self.frame_times))
    
    def draw_info_panel(self, frame, gesture):
        """Draw elegant info panel"""
        height, width = frame.shape[:2]
        
        if self.show_info:
            # Top-left panel
            panel_w, panel_h = 280, 160
            self.ui.draw_glassmorphism_panel(frame, 20, 20, panel_w, panel_h, 0.25)
            
            # FPS indicator
            fps_color = self.ui.SUCCESS_COLOR if self.fps >= 50 else self.ui.ACCENT_SECONDARY
            self.ui.draw_gradient_text(frame, f"FPS: {self.fps}", 40, 55, 0.8, 2)
            
            # FPS progress bar
            fps_progress = min(1.0, self.fps / 60.0)
            bar_x, bar_y = 40, 70
            bar_w, bar_h = 240, 6
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                         (60, 60, 70), -1, cv2.LINE_AA)
            
            # Progress bar with gradient
            progress_w = int(bar_w * fps_progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h),
                         fps_color, -1, cv2.LINE_AA)
            
            # Current gesture
            cv2.putText(frame, "GESTURE", (40, 105), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
            
            gesture_text = gesture['name']
            cv2.putText(frame, gesture_text, (40, 135), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, self.ui.TEXT_PRIMARY, 2, cv2.LINE_AA)
            
            # Finger count indicator with labels
            finger_count = gesture['finger_count']
            finger_names = ['👍', '☝️', '🖕', '💍', '🤙']
            dot_y = 160
            for i in range(5):
                is_extended = i < finger_count
                color = self.ui.ACCENT_PRIMARY if is_extended else (60, 60, 70)
                self.ui.draw_smooth_circle(frame, (50 + i*40, dot_y), 6, color, -1)
                
                # Show which fingers are detected
                if i in gesture.get('extended_fingers', []):
                    cv2.putText(frame, finger_names[i], (42 + i*40, dot_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ui.SUCCESS_COLOR, 1, cv2.LINE_AA)
            
            # 3D Animation mode (if enabled)
            if self.animation_3d and self.show_3d:
                mode_text = f"Mode: {self.animation_3d.mode}"
                obj_text = self.animation_3d.get_current_object_name()
                cv2.putText(frame, mode_text, (40, 195), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
                cv2.putText(frame, obj_text, (40, 215), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, self.ui.ACCENT_SECONDARY, 1, cv2.LINE_AA)
        
        # Gesture guide panel (right side)
        if self.show_guide:
            guide_w, guide_h = 300, 280
            guide_x = width - guide_w - 20
            self.ui.draw_glassmorphism_panel(frame, guide_x, 20, guide_w, guide_h, 0.25)
            
            # Title
            cv2.putText(frame, "GESTURE GUIDE", (guide_x + 20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui.TEXT_PRIMARY, 2, cv2.LINE_AA)
            
            # Gesture list
            gestures = [
                ("Swipe 💨", "Spin Object"),
                ("Pinch 👌", "Scale Size"),
                ("Fist ✊", "Stop/Freeze"),
                ("Point ☝", "Move Object"),
                ("Open 🖐", "Rotate")
            ]
            
            y_offset = 95
            for name, desc in gestures:
                cv2.putText(frame, name, (guide_x + 20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui.TEXT_PRIMARY, 1, cv2.LINE_AA)
                cv2.putText(frame, desc, (guide_x + 160, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
                y_offset += 35
                
            # Positioning Tips
            cv2.line(frame, (guide_x + 20, y_offset + 5), (guide_x + guide_w - 20, y_offset + 5), (100, 100, 120), 1)
            cv2.putText(frame, "TIPS:", (guide_x + 20, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui.ACCENT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(frame, "- Keep hand in center", (guide_x + 20, y_offset + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(frame, "- Swipe fully left/right", (guide_x + 20, y_offset + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
        
        # Bottom control hints
        controls = "ESC: Quit  |  H: Info  |  L: Landmarks  |  G: Guide  |  SPACE: Switch 3D"
        text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (width - text_size[0]) // 2
        
        # Background for controls
        self.ui.draw_glassmorphism_panel(frame, text_x - 10, height - 50,
                                        text_size[0] + 20, 35, 0.2)
        cv2.putText(frame, controls, (text_x, height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
    
    def draw_hand_visualization(self, frame, hand_landmarks, width, height, label='Left'):
        """Draw premium hand visualization with velocity arrow"""
        
        # Color Coding:
        # Left Hand (Rotate) -> Green/Cyan Theme
        # Right Hand (Scale) -> Orange/Blue Theme
        if label == 'Left':
            main_color = (100, 255, 100) # Green
            conn_color = (100, 255, 200)
        else:
            main_color = (100, 200, 255) # Orange/Yellowish
            conn_color = (100, 150, 255)
            
        if self.show_landmarks:
            # Custom landmark drawing with premium colors
            landmark_style = self.mp_drawing.DrawingSpec(
                color=main_color, 
                thickness=2,
                circle_radius=3
            )
            
            connection_style = self.mp_drawing.DrawingSpec(
                color=conn_color, 
                thickness=2
            )
            
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style
            )
        
        # Draw Velocity Vector (Arrow)
        # Showing the user how "fast" they are swiping
        if hasattr(self, 'hand_velocity'):
            speed = np.linalg.norm(self.hand_velocity)
            if speed > 2.0: # Only draw if moving
                # Get palm center
                wrist = hand_landmarks.landmark[0]
                middle = hand_landmarks.landmark[9]
                cx = int((wrist.x + middle.x) / 2 * width)
                cy = int((wrist.y + middle.y) / 2 * height)
                
                # Scale velocity for visualization
                end_x = int(cx + self.hand_velocity[0] * 3)
                end_y = int(cy + self.hand_velocity[1] * 3)
                
                # Color changes based on speed (Cyan -> Red)
                color_intensity = min(255, int(speed * 10))
                arrow_color = (100, 255 - color_intensity, 255)
                
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), arrow_color, 4, tipLength=0.3)
                
    def draw_center_marker(self, frame):
        """Draw subtle center marker for positioning guide"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Subtle crosshair
        size = 20
        color = (60, 60, 70)
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
        
        # "Best Position" text
        if self.show_guide:
            cv2.putText(frame, "Center Hand Here", (cx - 60, cy + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    
    def process_frame(self, frame):
        """Process frame with premium UI and 3D animations"""
        # Apply subtle vignette effect
        height, width = frame.shape[:2]
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Default gesture
        gesture = {
            'type': 'none',
            'name': 'No Hand',
            'finger_count': 0,
            'extended_fingers': [],
            'changed': False
        }
        
        # Draw persistent center marker
        self.draw_center_marker(frame)
        
        # Store frame for HUD drawing
        self.current_frame = frame
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            # Need to zip landmarks with classification
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get label (Left/Right)
                # MediaPipe treats 'Left' as the person's left hand (which appears on the right if not flipped, 
                # but we flipped the frame). 
                # After cv2.flip(frame, 1):
                # - Physical Right Hand -> Appears on Right Side -> Labeled as "Right"
                # - Physical Left Hand -> Appears on Left Side -> Labeled as "Left"
                label = handedness.classification[0].label
                
                # Draw hand visualization (Pass label for color coding)
                self.draw_hand_visualization(frame, hand_landmarks, width, height, label)
                
                # Detect gesture
                landmarks = hand_landmarks.landmark
                gesture = self.gesture_detector.detect_gesture(landmarks)
                
                # Get hand positions for 3D control
                fingertip_positions = self.gesture_detector.get_fingertip_positions(
                    landmarks, width, height
                )
                palm_center = self.gesture_detector.get_palm_center(
                    landmarks, width, height
                )
                
                # Handle 3D animation and particles based on gesture AND handedness
                if self.animation_3d and self.show_3d:
                    self.handle_gesture_effects(gesture, landmarks, fingertip_positions, palm_center, label)
        
        # Update 3D animation
        if self.animation_3d and self.show_3d:
            current_time = time.time()
            delta_time = current_time - self.prev_time
            self.prev_time = current_time
            
            self.animation_3d.update(delta_time)
            
        # ALWAYS draw 3D object
        if self.animation_3d and self.show_3d:
            self.animation_3d.draw(frame)
        
        # Draw UI overlays
        self.draw_info_panel(frame, gesture)
        
        # Update FPS
        self.update_fps()
        
        return frame
    
    print("DEBUG: handle_gesture_effects called")
    def handle_gesture_effects(self, gesture, landmarks, fingertip_positions, palm_center, handedness_label):
        """
        Handle gesture-based effects with STRICT DUAL HAND roles:
        - LEFT Hand: Rotation Control ONLY (Grab to Rotate)
        - RIGHT Hand: Scale Control ONLY (Pinch to Scale)
        """
        gesture_type = gesture['type']
        
        # Get hand orientation
        pitch, yaw, roll = self.gesture_detector.get_hand_orientation(landmarks)
        hand_rotation = np.array([pitch, yaw, roll])
        
        # Check Pinch Distance (Thumb to Index)
        pinch_dist = self.gesture_detector.get_pinch_distance(landmarks, self.animation_3d.width, self.animation_3d.height)
        
        # --- LEFT HAND: ROTATION CONTROL ---
        if handedness_label == 'Left':
            
            # Draw Mode Indicator
            self.ui.draw_hand_hud(self.current_frame, palm_center, "ROTATION", (100, 255, 100))
            
            # GRAB LOGIC (PINCH) -> ONLY FOR ROTATION
            is_pinching = pinch_dist is not None and pinch_dist < 0.05
            
            if is_pinching:
                # -- STATE: GRABBING (ROTATION LOCKED) --
                if not self.animation_3d.is_grabbed:
                    self.animation_3d.is_grabbed = True
                    self.animation_3d.set_mode('GRAB')
                    
                    # Lock Offset
                    self.animation_3d.grab_start_hand_rotation = hand_rotation.copy()
                    self.animation_3d.grab_start_object_rotation = self.animation_3d.current_rotation.copy()
                    print("🔒 LEFT HAND GRAB - Locked Rotation")
                
                # Draw Lock UI
                self.ui.draw_hand_hud(self.current_frame, palm_center, "LOCKED", (100, 255, 100))
                
                # Update Object Rotation based on Hand Rotation + Offset
                # New Object Rot = Start Object Rot + (Current Hand Rot - Start Hand Rot)
                rotation_delta = hand_rotation - self.animation_3d.grab_start_hand_rotation
                
                # Sensitivity 1.2 for slightly amplified feel but consistent
                sensitivity = 1.2
                target_rot = self.animation_3d.grab_start_object_rotation + rotation_delta * sensitivity
                
                # Apply Smoothing to remove jitter (Make it feel 'heavy' yet responsive)
                # alpha = 0.2 means tight control but filters high-frequency noise
                alpha = 0.2
                diff = target_rot - self.animation_3d.current_rotation
                self.animation_3d.current_rotation += diff * alpha
                self.animation_3d.target_rotation = self.animation_3d.current_rotation.copy()
                
                # Kill velocity
                self.animation_3d.angular_velocity *= 0.0
                
            else:
                # -- STATE: RELEASED --
                if self.animation_3d.is_grabbed:
                    self.animation_3d.is_grabbed = False
                    self.animation_3d.set_mode('INERTIA')
                    print("🔓 LEFT HAND RELEASED - Spin")
                    
                    # Calculate Throw Velocity
                    current_pos = np.array([palm_center[0], palm_center[1]])
                    if self.prev_palm_center is not None:
                        # Simple velocity
                        velocity = current_pos - self.prev_palm_center
                        spin_power = 0.5
                        # Invert Y for natural feel
                        rot_velocity = [-velocity[1] * spin_power, velocity[0] * spin_power, 0]
                        self.animation_3d.apply_impulse(rot_velocity)
                
                # Update velocity for next frame (always track for throw)
                current_pos = np.array([palm_center[0], palm_center[1]])
                self.prev_palm_center = current_pos

                # Fist Gesture for Stop (Only Left Hand stops rotation)
                if gesture_type == 'fist':
                     self.animation_3d.set_mode('FREEZE')
                     self.ui.draw_hand_hud(self.current_frame, palm_center, "STOP", (255, 50, 50))

        # --- RIGHT HAND: SCALE CONTROL ---
        elif handedness_label == 'Right':
            
            # Draw Mode Indicator
            self.ui.draw_hand_hud(self.current_frame, palm_center, "SCALE", (100, 200, 255))
            
            # Pinch Logic -> ONLY FOR SCALE
            if pinch_dist is not None:
                # Map distance to scale
                # Normal range 0.03 (closed) to 0.25 (open)
                target_scale = np.interp(pinch_dist, [0.03, 0.25], [80, 400])
                current_scale = self.animation_3d.objects[self.animation_3d.current_object_idx].scale
                
                # Apply smooth scaling
                new_scale = current_scale * 0.9 + target_scale * 0.1
                self.animation_3d.objects[self.animation_3d.current_object_idx].scale = new_scale
            
            # Right Hand Gestures
            if gesture_type == 'pointing':
                self.animation_3d.set_mode('FOLLOW')
                if len(fingertip_positions) > 1:
                    self.animation_3d.update_follow(fingertip_positions[1])
            
            elif gesture_type == 'peace':
                self.animation_3d.set_mode('ORBIT')

    
    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return
        
        print("🚀 Premium Hand Gesture App Started")
        print("📹 Show your hand to the camera")
        
        # Create named window with specific properties
        cv2.namedWindow('Premium Hand Gesture Recognition', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Premium Hand Gesture Recognition', processed_frame)
                
                # NO DELAY - Max Speed
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or Q
                    print("👋 Exiting...")
                    break
                elif key == ord('h'):
                    self.show_info = not self.show_info
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                elif key == ord('g'):
                    self.show_guide = not self.show_guide
                elif key == ord(' '):  # SPACE - switch 3D object
                    if self.animation_3d:
                        self.animation_3d.switch_object()
                        print(f"🎨 Switched to: {self.animation_3d.get_current_object_name()}")
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("✅ Cleanup complete")


def main():
    """Entry point"""
    print("=" * 60)
    print("✨ Premium Hand Gesture Recognition UI ✨")
    print("=" * 60)
    
    try:
        app = HandGestureApp()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
