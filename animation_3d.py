"""
3D Animation System with Gesture Control
Renders 3D objects that can be controlled by hand gestures
"""
import numpy as np
import cv2
import math
import time


class Object3D:
    """3D object with vertices, rotation, and position"""
    
    def __init__(self, vertices, edges, name="Object"):
        self.vertices = np.array(vertices, dtype=float)
        self.edges = edges
        self.name = name
        self.rotation = np.array([0.0, 0.0, 0.0])  # x, y, z rotation
        self.position = np.array([0.0, 0.0, 0.0])
        self.scale = 100.0
        
    @staticmethod
    def create_cube():
        """Create a cube"""
        vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        return Object3D(vertices, edges, "Cube")
    
    @staticmethod
    def create_pyramid():
        """Create a pyramid"""
        vertices = [
            [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1],
            [0, 1.5, 0]
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 4), (2, 4), (3, 4)
        ]
        return Object3D(vertices, edges, "Pyramid")
    
    @staticmethod
    def create_sphere(segments=12):
        """Create a wireframe sphere"""
        vertices = []
        edges = []
        
        # Generate vertices
        for i in range(segments + 1):
            lat = (i * math.pi) / segments - math.pi / 2
            for j in range(segments):
                lon = (j * 2 * math.pi) / segments
                x = math.cos(lat) * math.cos(lon)
                y = math.sin(lat)
                z = math.cos(lat) * math.sin(lon)
                vertices.append([x, y, z])
        
        # Generate edges
        for i in range(segments):
            for j in range(segments):
                p1 = i * segments + j
                p2 = i * segments + (j + 1) % segments
                p3 = (i + 1) * segments + j
                edges.append((p1, p2))
                edges.append((p1, p3))
        
        return Object3D(vertices, edges, "Sphere")


class Animation3D:
    """Main 3D animation controller"""
    
    MODES = {
        'FREEZE': 0,
        'FOLLOW': 1,
        'ORBIT': 2,
        'PARTICLE': 3,
        'FREE_ROTATION': 4
    }
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.center = np.array([width // 2, height // 2])
        
        # 3D Objects
        self.objects = [
            Object3D.create_cube(),
            Object3D.create_pyramid(),
            Object3D.create_sphere(10)
        ]
        self.current_object_idx = 0
        
        # Make objects bigger and more visible
        for obj in self.objects:
            obj.scale = 200.0  # Even bigger - was 150
        
        # Animation state
        self.mode = 'FREE_ROTATION'
        self.auto_rotation_angle = 0
        self.target_rotation = np.array([0.3, 0.5, 0.0])  # Better initial angle
        self.current_rotation = np.array([0.3, 0.5, 0.0])
        
        # Physics state for Inertial Rotation
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.friction = 0.95  # Slow down factor (0.0 to 1.0)
        
        # Grab State for Direct Manipulation
        self.is_grabbed = False
        self.grab_start_hand_rotation = np.array([0.0, 0.0, 0.0])
        self.grab_start_object_rotation = np.array([0.0, 0.0, 0.0])
        
        # Smoothing
        self.smooth_factor = 0.2  # Slightly faster response
        
        # Colors (BGR) - Brighter and more visible
        self.primary_color = (220, 150, 255)  # Brighter Purple
        self.secondary_color = (255, 220, 120)  # Bright Cyan/Yellow
        self.accent_color = (120, 255, 120)  # Bright Green
        
    def get_rotation_matrix(self, angles):
        """Calculate 3D rotation matrix"""
        rx, ry, rz = angles
        
        # Rotation around X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Rotation around Y
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Rotation around Z
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def apply_impulse(self, velocity):
        """Apply rotational impulse (add velocity)"""
        self.angular_velocity += np.array(velocity) * 0.1

    def project_to_2d(self, vertices, scale, center_offset):
        """Project 3D vertices to 2D screen coordinates"""
        # Apply rotation
        rotation_matrix = self.get_rotation_matrix(self.current_rotation)
        rotated = vertices @ rotation_matrix.T
        
        # Simple perspective projection with better visibility
        projected = []
        for vertex in rotated:
            x, y, z = vertex
            
            # Simple scaling without complex perspective
            # This ensures the object is always visible
            x_proj = x * scale + self.center[0] + center_offset[0]
            y_proj = y * scale + self.center[1] + center_offset[1]
            
            projected.append([x_proj, y_proj, z])
        
        return np.array(projected)
    
    def set_mode(self, mode):
        """Set animation mode"""
        if mode in self.MODES:
            self.mode = mode
    
    def update_free_rotation(self, hand_pitch, hand_yaw, hand_roll=0):
        """Update rotation based on hand orientation"""
        self.target_rotation[0] = hand_pitch
        self.target_rotation[1] = hand_yaw
        self.target_rotation[2] = hand_roll
    
    def update_orbit(self, delta_time):
        """Auto-rotation orbit mode"""
        self.auto_rotation_angle += delta_time * 2.0
        self.target_rotation[1] = self.auto_rotation_angle
        self.target_rotation[0] = math.sin(self.auto_rotation_angle * 0.5) * 0.3
    
    def update_follow(self, fingertip_pos):
        """Follow fingertip position"""
        # Convert fingertip position to center offset
        offset_x = fingertip_pos[0] - self.center[0]
        offset_y = fingertip_pos[1] - self.center[1]
        
        # Map to rotation
        self.target_rotation[1] = (offset_x / self.width) * math.pi
        self.target_rotation[0] = (offset_y / self.height) * math.pi
    
    def update(self, delta_time):
        """Update animation state with physics"""
        
        if self.mode == 'INERTIA':
            # Physics mode: Apply velocity and friction
            self.current_rotation += self.angular_velocity * delta_time
            self.angular_velocity *= self.friction # Decay
            self.target_rotation = self.current_rotation.copy()
            
        elif self.mode == 'FREE_ROTATION':
            # Manual control mode: Smooth interpolation to target
            # Still use smoothing alpha
            alpha = 0.1
            diff = self.target_rotation - self.current_rotation
            self.current_rotation += diff * alpha
            # Kill velocity when manual controlling
            self.angular_velocity *= 0.0

        elif self.mode == 'ORBIT':
            # Orbit mode
            self.current_rotation[1] += 2.0 * delta_time
            self.current_rotation[0] = 0.3 # Fixed tilt
            
        elif self.mode == 'GRAB':
            # Direct Manipulation: Rotation is handled externally by calculating offset
            # Just apply minimal smoothing to remove jitter
            # But the MAIN logic for setting current_rotation happens in the gesture handler
            pass
            
        elif self.mode == 'FREEZE':
            # Stop
            self.angular_velocity *= 0.0
            
        elif self.mode == 'FOLLOW':
             # Just rotate slowly
             self.current_rotation[1] += 0.5 * delta_time

    
    def draw(self, frame, center_offset=(0, 0)):
        """Draw 3D object on frame"""
        obj = self.objects[self.current_object_idx]
        
        # Project vertices
        projected = self.project_to_2d(obj.vertices, obj.scale, center_offset)
        
        # Sort edges by depth (painter's algorithm)
        edge_depths = []
        for edge in obj.edges:
            depth = (projected[edge[0]][2] + projected[edge[1]][2]) / 2
            edge_depths.append((edge, depth))
        
        edge_depths.sort(key=lambda x: x[1], reverse=True)
        
        # Draw edges with depth-based coloring
        for (start_idx, end_idx), depth in edge_depths:
            p1 = tuple(projected[start_idx][:2].astype(int))
            p2 = tuple(projected[end_idx][:2].astype(int))
            
            # Color based on depth - more visible
            depth_factor = max(0, min(1, (depth + 2) / 4))
            
            # Choose color based on mode
            if self.mode == 'ORBIT':
                base_color = self.secondary_color
            elif self.mode == 'FOLLOW':
                base_color = self.accent_color
            else:
                base_color = self.primary_color
            
            # Apply depth shading (keep it bright)
            shaded_color = tuple(int(c * (0.5 + depth_factor * 0.5)) for c in base_color)
            
            # Draw thick glow effect first
            cv2.line(frame, p1, p2, shaded_color, 8, cv2.LINE_AA)
            
            # Draw bright main line
            cv2.line(frame, p1, p2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Draw vertices - bigger and brighter
        for point in projected:
            pos = tuple(point[:2].astype(int))
            # Outer glow
            cv2.circle(frame, pos, 8, base_color, -1, cv2.LINE_AA)
            # Inner bright dot
            cv2.circle(frame, pos, 5, (255, 255, 255), -1, cv2.LINE_AA)
    
    def switch_object(self):
        """Switch to next 3D object"""
        self.current_object_idx = (self.current_object_idx + 1) % len(self.objects)
    
    def get_current_object_name(self):
        """Get name of current object"""
        return self.objects[self.current_object_idx].name
