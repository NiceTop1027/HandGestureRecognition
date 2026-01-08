"""
Hand Gesture Recognition - Premium UI
Minimalist, high-performance hand tracking application with Holographic AR

Controls:
- ESC or Q: Quit application
- H: Toggle info panel
- L: Toggle hand landmarks
- G: Toggle gesture guide
- SPACE: Switch 3D Object
- TAB: Switch Camera
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame 
from gesture_detector import GestureDetector
from animation_3d import Animation3D
from particle_system import ParticleSystem
from text_renderer import TextRenderer
from ai_translator import GeminiTranslator
from renderer_opengl import RendererGL # New OpenGL Renderer

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
        self.text_renderer = TextRenderer()
        
    def draw_glassmorphism_panel(self, frame, x, y, w, h, alpha=0.2):
        """Draw glassmorphism panel with ROI optimization"""
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
        cv2.putText(frame, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
        
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, self.TEXT_PRIMARY, thickness, cv2.LINE_AA)

    def draw_korean_text(self, frame, text, x, y, color=(255, 255, 255)):
        """Draw Korean text using TextRenderer"""
        frame[:] = self.text_renderer.put_text(frame, text, x, y, color)
    
    def draw_smooth_circle(self, frame, center, radius, color, thickness=-1):
        """Draw anti-aliased circle"""
        cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
    
    def draw_hand_hud(self, frame, center, text, color):
        """Draw futuristic HUD around hand"""
        x, y = center
        
        # Rotating rings animation
        self.animation_time += 0.2
        radius = 50 + int(np.sin(self.animation_time) * 5)
        
        # Outer ring
        cv2.circle(frame, (x, y), radius, color, 1, cv2.LINE_AA)
        
        # Inner segmented ring
        offset = self.animation_time * 20
        for i in range(0, 360, 60):
            angle = np.radians(i + offset)
            end_x = int(x + np.cos(angle) * (radius - 10))
            end_y = int(y + np.sin(angle) * (radius - 10))
            cv2.circle(frame, (end_x, end_y), 3, color, -1)
            
        # Label
        cv2.putText(frame, text, (x - 30, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Connection line to center
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(frame, (x, y), (cx, cy), color, 1, cv2.LINE_AA)
        
        # Data point
        mix = (self.animation_time % 5) / 5.0
        dot_x = int(x + (cx - x) * mix)
        dot_y = int(y + (cy - y) * mix)
        cv2.circle(frame, (dot_x, dot_y), 4, (255, 255, 255), -1)

class HandGestureApp:
    """Main application with premium UI and OpenGL AR"""
    
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Components
        self.gesture_detector = GestureDetector()
        self.ui = PremiumUI()
        
        # Will be initialized in run()
        self.animation_3d = None 
        self.renderer = None
        
        # AI Translation State
        try:
            from keys import GEMINI_API_KEY
            self.translator = GeminiTranslator(GEMINI_API_KEY)
        except ImportError:
            print("⚠️ keys.py not found or GEMINI_API_KEY missing. AI Translation disabled.")
            self.translator = GeminiTranslator(None)
            
        self.gesture_buffer = []
        self.last_buffer_update = time.time()
        self.current_translation = ""
        self.is_translating = False
        
        # UI state
        self.show_info = True
        self.show_landmarks = True
        self.show_guide = False
        self.show_3d = True
        
        # Performance tracking
        self.fps = 0
        self.frame_times = []
        self.last_time = time.time()
        
        # Gesture control state
        self.prev_palm_center = None
        self.hand_velocity = np.array([0.0, 0.0])
        
        self.cap = None
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera and renderer"""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(camera_index)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        if not self.cap.isOpened():
            print(f"⚠️ Camera index {camera_index} not available, trying 0")
            if camera_index != 0:
                return self.initialize_camera(0)
            return False
            
        self.current_camera_index = camera_index
        
        # Get dimensions
        ret, frame = self.cap.read()
        if ret:
            height, width = frame.shape[:2]
            # Initialize Physics/Logic Engine if not already done
            if self.animation_3d is None:
                self.animation_3d = Animation3D(width, height)
            # Initialize OpenGL Renderer if not already done
            if self.renderer is None:
                self.renderer = RendererGL(width, height)
        
        print(f"✅ Camera {camera_index} initialized")
        return True
    
    def switch_camera(self):
        """Switch to the next available camera"""
        next_index = self.current_camera_index + 1
        # Try up to index 4, then wrap back to 0
        if next_index > 4: 
            next_index = 0
            
        print(f"🔄 Switching to camera {next_index}...")
        
        # Attempt to initialize next camera
        # We temporarily create a new capture to check if valid
        temp_cap = cv2.VideoCapture(next_index)
        if temp_cap.isOpened():
            temp_cap.release()
            self.initialize_camera(next_index)
        else:
            print(f"⚠️ Camera {next_index} not found. Wrapping to 0.")
            self.initialize_camera(0)

    def update_fps(self):
        """Calculate smooth FPS"""
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        if delta > 0:
            self.frame_times.append(1.0 / delta)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.fps = int(sum(self.frame_times) / len(self.frame_times))
    
    def draw_info_panel(self, frame, gesture):
        """Draw elegant info panel"""
        if not self.show_info: return
        
        height, width = frame.shape[:2]
        
        # Main Panel
        panel_w, panel_h = 280, 160
        self.ui.draw_glassmorphism_panel(frame, 20, 20, panel_w, panel_h, 0.25)
        
        # FPS
        fps_color = self.ui.SUCCESS_COLOR if self.fps >= 50 else self.ui.ACCENT_SECONDARY
        self.ui.draw_gradient_text(frame, f"FPS: {self.fps}", 40, 55, 0.8, 2)
        
        # FPS Bar
        fps_progress = min(1.0, self.fps / 60.0)
        cv2.rectangle(frame, (40, 70), (280, 76), (60, 60, 70), -1)
        cv2.rectangle(frame, (40, 70), (40 + int(240 * fps_progress), 76), fps_color, -1)
        
        # Gesture Info
        cv2.putText(frame, "GESTURE", (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.putText(frame, gesture['name'], (40, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui.TEXT_PRIMARY, 2, cv2.LINE_AA)
        
        # Finger dots
        finger_count = gesture['finger_count']
        dot_y = 160
        for i in range(5):
            is_extended = i < finger_count
            color = self.ui.ACCENT_PRIMARY if is_extended else (60, 60, 70)
            self.ui.draw_smooth_circle(frame, (50 + i*40, dot_y), 6, color, -1)

        # 3D Info
        if self.animation_3d and self.show_3d:
            mode_text = f"Mode: {self.animation_3d.mode}"
            obj_text = self.animation_3d.get_current_object_name()
            cv2.putText(frame, mode_text, (40, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)
            cv2.putText(frame, obj_text, (40, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui.ACCENT_SECONDARY, 1, cv2.LINE_AA)

        # AI Translation Display
        if self.current_translation:
             self.ui.draw_glassmorphism_panel(frame, 20, height - 100, width - 40, 80, 0.3)
             self.ui.draw_korean_text(frame, self.current_translation, 40, height - 50, self.ui.SUCCESS_COLOR)

        if self.is_translating:
             cv2.putText(frame, "Translating...", (width-200, height-95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 1, cv2.LINE_AA)
        
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
        controls = "ESC: Quit | TAB: Camera | SPACE: Object | H: Info | L: Landmarks"
        text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (width - text_size[0]) // 2
        
        # Background for controls
        self.ui.draw_glassmorphism_panel(frame, text_x - 10, height - 50,
                                        text_size[0] + 20, 35, 0.2)
        cv2.putText(frame, controls, (text_x, height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui.TEXT_SECONDARY, 1, cv2.LINE_AA)

    def process_frame(self, frame):
        """Process frame (Gestures + UI) - Returns 2D frame"""
        height, width = frame.shape[:2]
        frame = cv2.flip(frame, 1) # Mirror
        
        # MediaPipe Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = {'type': 'none', 'name': 'No Hand', 'finger_count': 0}
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                
                # Draw Landmarks (2D Overlay)
                if self.show_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect Gesture & Update 3D Logic
                gesture = self.gesture_detector.detect_gesture(hand_landmarks.landmark)
                
                # Hand Stats
                palm_center = self.gesture_detector.get_palm_center(hand_landmarks.landmark, width, height)
                fingertip_positions = self.gesture_detector.get_fingertip_positions(hand_landmarks.landmark, width, height)
                
                # AI Prediction Buffer
                predicted = self.gesture_detector.predict_gesture(hand_landmarks.landmark)
                if predicted and predicted != "?":
                    current_time = time.time()
                    if current_time - self.last_buffer_update > 2.0:
                        if not self.gesture_buffer or self.gesture_buffer[-1] != predicted:
                            self.gesture_buffer.append(predicted)
                            self.last_buffer_update = current_time
                
                # --- 3D Interaction Logic ---
                if self.animation_3d and self.show_3d:
                    self.handle_interaction(gesture, hand_landmarks.landmark, palm_center, fingertip_positions, label)
        
        # Update Physics
        if self.animation_3d:
            self.animation_3d.update(0.016) # Assume 60fps for physics step

        # Draw UI Overlay (2D)
        if self.show_info:
            self.draw_info_panel(frame, gesture)
            
        self.update_fps()
        return frame

    def handle_interaction(self, gesture, landmarks, palm_center, fingertips, label):
        """Handle gesture interaction for 3D object"""
        # Calculate pinch for interactions
        pinch_dist = self.gesture_detector.get_pinch_distance(landmarks, 1280, 720)
        
        # Calculate rotation from hand orientation
        pitch, yaw, roll = self.gesture_detector.get_hand_orientation(landmarks)
        hand_rotation = np.array([pitch, yaw, roll])

        # LEFT HAND: Rotate
        if label == 'Left':
            is_pinching = pinch_dist is not None and pinch_dist < 0.05
            
            if is_pinching:
                if not self.animation_3d.is_grabbed:
                    self.animation_3d.is_grabbed = True
                    self.animation_3d.set_mode('GRAB')
                    self.animation_3d.grab_start_hand_rotation = hand_rotation.copy()
                    self.animation_3d.grab_start_object_rotation = self.animation_3d.current_rotation.copy()
                    self.animation_3d.rotation_history = []  # Reset history on new grab
                
                # Update Rotation
                rot_delta = hand_rotation - self.animation_3d.grab_start_hand_rotation
                new_rotation = self.animation_3d.grab_start_object_rotation + rot_delta * 1.5
                
                # Track rotation change for momentum
                rotation_change = new_rotation - self.animation_3d.current_rotation
                self.animation_3d.rotation_history.append(rotation_change)
                
                # Keep only last 5 frames
                if len(self.animation_3d.rotation_history) > 5:
                    self.animation_3d.rotation_history.pop(0)
                
                self.animation_3d.target_rotation = new_rotation
                self.animation_3d.current_rotation += (self.animation_3d.target_rotation - self.animation_3d.current_rotation) * 0.2
            else:
                # Just released?
                if self.animation_3d.is_grabbed:
                    # Calculate momentum from recent rotation changes
                    if len(self.animation_3d.rotation_history) > 0:
                        avg_change = np.mean(self.animation_3d.rotation_history, axis=0)
                        # Apply momentum (boost it for better feel)
                        self.animation_3d.angular_velocity = avg_change * 3.0
                    
                    self.animation_3d.is_grabbed = False
                    self.animation_3d.set_mode('INERTIA')
                
                # Fist gesture freezes
                if gesture['type'] == 'fist':
                     self.animation_3d.set_mode('FREEZE')

        # RIGHT HAND: Scale
        elif label == 'Right':
            if pinch_dist is not None:
                # Map pinch to scale (50 to 400)
                target_scale = np.interp(pinch_dist, [0.03, 0.25], [50, 400])
                current_obj = self.animation_3d.objects[self.animation_3d.current_object_idx]
                current_obj.scale = current_obj.scale * 0.9 + target_scale * 0.1



    def run(self):
        """Main Loop"""
        if not self.initialize_camera(): return

        print("🚀 Holographic AR Started")
        
        running = True
        try:
            while running:
                # 1. Capture Frame
                ret, frame = self.cap.read()
                if not ret: break
                
                # 2. Process (Tracking + 2D UI)
                # processed_frame has UI drawn on it
                processed_frame = self.process_frame(frame)
                
                # 3. Create a separate UI overlay if needed, 
                # but simplified: Pass processed_frame as background to OpenGL
                # The 3D object will be drawn ON TOP of this background.
                # However, our UI (text) is drawn on processed_frame. 
                # So: Background -> Camera+2D UI. Foreground -> 3D Object.
                # Ideally: Background -> Camera. Middle -> 3D Object. Top -> UI.
                # For simplicity/performance now: Background = Camera+UI, then 3D on top.
                # This might occlude UI if 3D is big. 
                # FIX: Draw UI *after* separately?
                # Let's do: Background = Camera. 3D Object. UI Overlay.
                # RendererGL supports UI overlay texture.
                
                # Recalculate approach for best visuals:
                # Step A: Clean Camera Frame -> Render as GL Background
                # Step B: Render 3D Object in GL
                # Step C: Create Transparent UI Layer (OpenCV) -> Render as GL Foreground
                
                # Create clean UI layer
                ui_layer = np.zeros((720, 1280, 4), dtype=np.uint8) 
                # (Can't easily draw OpenCV on RGBA surface efficiently every frame without overhead)
                # STICK TO: Background (Camera) -> 3D -> UI (drawn on top of camera frame) ?
                # If we draw UI on camera frame, then 3D object covers UI.
                # If we want UI on top, we need to pass it separately.
                
                # FAST PATH: 
                # 1. Get Camera Frame.
                # 2. Draw 2D UI on it (in-place). 
                # 3. Render this as background.
                # 4. Render 3D Object (Hologram).
                # Result: Hologram is ON TOP of UI. This is actually "Cinematic" (UI is on screen, Hologram is projected in space).
                # If UI gets blocked, we can adjust.
                
                # Get State for Renderer
                rot_angles = self.animation_3d.current_rotation
                scale = self.animation_3d.objects[self.animation_3d.current_object_idx].scale
                shape = self.animation_3d.get_current_object_name()
                
                # Render Scene (Returns Pygame Keys)
                keys = self.renderer.render_scene(processed_frame, rot_angles, scale, shape_type=shape)
                self.renderer.flip()
                
                # Handle Inputs
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_h:
                            self.show_info = not self.show_info
                        elif event.key == pygame.K_l:
                            self.show_landmarks = not self.show_landmarks
                        elif event.key == pygame.K_g:
                            self.show_guide = not self.show_guide
                        elif event.key == pygame.K_SPACE:
                            self.animation_3d.switch_object()
                        elif event.key == pygame.K_TAB: # Switch Camera
                            self.switch_camera()
                        elif event.key == pygame.K_t: # Translate
                             if self.gesture_buffer:
                                 self.current_translation = self.translator.translate(self.gesture_buffer)
                                 self.gesture_buffer = []

        finally:
            self.renderer.quit()
            self.cap.release()
            self.hands.close()

if __name__ == "__main__":
    HandGestureApp().run()
