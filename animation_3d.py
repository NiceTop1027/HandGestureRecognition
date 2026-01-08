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
    
    def __init__(self, vertices, edges, faces=None, name="Object"):
        self.vertices = np.array(vertices, dtype=float)
        self.edges = edges
        self.faces = faces if faces is not None else []
        self.name = name
        self.rotation = np.array([0.0, 0.0, 0.0])  # x, y, z rotation
        self.position = np.array([0.0, 0.0, 0.0])
        self.scale = 100.0
    
    def get_face_center(self, face_idx):
        """Get center point of a face in local 3D space"""
        if face_idx >= len(self.faces):
            return np.array([0.0, 0.0, 0.0])
        
        face_vertices = [self.vertices[i] for i in self.faces[face_idx]]
        return np.mean(face_vertices, axis=0)
    
    def get_face_normal(self, face_idx):
        """Get normal vector of a face"""
        if face_idx >= len(self.faces) or len(self.faces[face_idx]) < 3:
            return np.array([0.0, 0.0, 1.0])
        
        # Get three vertices of the face
        v0 = self.vertices[self.faces[face_idx][0]]
        v1 = self.vertices[self.faces[face_idx][1]]
        v2 = self.vertices[self.faces[face_idx][2]]
        
        # Calculate normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Normalize
        length = np.linalg.norm(normal)
        if length > 0:
            normal = normal / length
        
        return normal
    
    def extrude_face(self, face_idx, distance):
        """Extrude a face to create new geometry"""
        if face_idx >= len(self.faces):
            return
        
        face = self.faces[face_idx]
        normal = self.get_face_normal(face_idx)
        
        # Create new vertices by offsetting the face vertices
        new_vertex_indices = []
        num_original_vertices = len(self.vertices)
        
        for vi in face:
            new_vertex = self.vertices[vi] + normal * distance
            self.vertices = np.vstack([self.vertices, new_vertex])
            new_vertex_indices.append(num_original_vertices + len(new_vertex_indices))
        
        # Create connecting edges between old and new vertices
        for i in range(len(face)):
            old_v = face[i]
            new_v = new_vertex_indices[i]
            # Vertical edge
            self.edges.append((old_v, new_v))
            
            # Horizontal edge on new face
            next_i = (i + 1) % len(face)
            self.edges.append((new_v, new_vertex_indices[next_i]))
        
        # Add new face at the extruded position
        self.faces.append(new_vertex_indices)
        
        return new_vertex_indices
        
    @staticmethod
    def create_cube():
        """Create a cube"""
        vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Front: 0,1,2,3
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]        # Back: 4,5,6,7
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        # Define 6 faces (each face is a list of 4 vertex indices in counter-clockwise order)
        faces = [
            [0, 1, 2, 3],  # Front (z = -1)
            [4, 7, 6, 5],  # Back (z = 1)
            [0, 4, 5, 1],  # Bottom (y = -1)
            [3, 2, 6, 7],  # Top (y = 1)
            [0, 3, 7, 4],  # Left (x = -1)
            [1, 5, 6, 2],  # Right (x = 1)
        ]
        return Object3D(vertices, edges, faces, "Cube")
    
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
        self.friction = 0.92  # Slow down factor (lower = more friction)
        
        # Grab State for Direct Manipulation
        self.is_grabbed = False
        self.grab_start_hand_rotation = np.array([0.0, 0.0, 0.0])
        self.grab_start_object_rotation = np.array([0.0, 0.0, 0.0])
        
        # Momentum tracking for smooth release
        self.last_rotation = self.current_rotation.copy()
        self.rotation_history = []  # Track recent rotation changes
        
        # Face selection and extrusion
        self.selected_face_idx = None
        self.is_extruding = False
        self.extrusion_start_pos = None
        self.extrusion_distance = 0.0
        
        # Smoothing
        self.smooth_factor = 0.2  # Slightly faster response
        
        # Colors (BGR) - Brighter and more visible
        self.primary_color = (220, 150, 255)  # Brighter Purple
        self.secondary_color = (255, 220, 120)  # Bright Cyan/Yellow
        self.accent_color = (120, 255, 120)  # Bright Green
    
    def select_face_by_position(self, hand_x, hand_y, projected_vertices):
        """Select closest face to hand position"""
        obj = self.objects[self.current_object_idx]
        
        if not obj.faces:
            return None
        
        min_dist = float('inf')
        closest_face = None
        
        for i, face in enumerate(obj.faces):
            # Skip if any vertex index is out of bounds
            if any(vi >= len(projected_vertices) for vi in face):
                continue
                
            try:
                # Calculate face center in 2D screen space
                face_center_2d = np.mean([projected_vertices[vi][:2] for vi in face], axis=0)
                
                # Distance to hand
                dist = np.linalg.norm(face_center_2d - np.array([hand_x, hand_y]))
                
                if dist < min_dist:
                    min_dist = dist
                    closest_face = i
            except (IndexError, TypeError):
                # Skip invalid faces
                continue
        
        # Only select if reasonably close (within 150 pixels)
        if closest_face is not None and min_dist < 150:
            return closest_face
        return None
        
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
            self.current_rotation += self.angular_velocity
            self.angular_velocity *= self.friction
            
            # Stop if velocity is very small
            if np.linalg.norm(self.angular_velocity) < 0.001:
                self.angular_velocity = np.array([0.0, 0.0, 0.0])
            
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
        
        # Highlight selected face if in extrusion mode
        if self.selected_face_idx is not None and obj.faces:
            face = obj.faces[self.selected_face_idx]
            
            # Draw face outline in bright color
            face_points = [tuple(projected[vi][:2].astype(int)) for vi in face]
            
            # Fill face with semi-transparent highlight
            overlay = frame.copy()
            pts = np.array(face_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 255))  # Cyan highlight
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw bright outline
            for i in range(len(face_points)):
                p1 = face_points[i]
                p2 = face_points[(i + 1) % len(face_points)]
                cv2.line(frame, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
            
            # Show extrusion preview if actively extruding
            if self.is_extruding and self.extrusion_distance > 10:
                # Get face normal
                normal = obj.get_face_normal(self.selected_face_idx)
                
                # Apply rotation to normal
                rotation_matrix = self.get_rotation_matrix(self.current_rotation)
                rotated_normal = rotation_matrix @ normal
                
                # Calculate extrusion offset (simplified: use distance / 2 as depth)
                extrude_amount = self.extrusion_distance / 200.0  # Scale factor
                offset_3d = rotated_normal * extrude_amount
                
                # Project extruded face points
                extruded_vertices = []
                for vi in face:
                    new_vertex = obj.vertices[vi] + offset_3d
                    extruded_vertices.append(new_vertex)
                
                # Project to 2D
                extruded_projected = []
                for vertex in extruded_vertices:
                    rotated = rotation_matrix @ vertex
                    x = rotated[0] * obj.scale + self.center[0] + center_offset[0]
                    y = rotated[1] * obj.scale + self.center[1] + center_offset[1]
                    extruded_projected.append((int(x), int(y)))
                
                # Draw extruded face preview (semi-transparent)
                overlay = frame.copy()
                pts = np.array(extruded_projected, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (255, 100, 0))  # Orange preview
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                # Draw connecting lines
                for i in range(len(face_points)):
                    cv2.line(frame, face_points[i], extruded_projected[i], (255, 200, 0), 2, cv2.LINE_AA)
    
    def switch_object(self):
        """Switch to next 3D object"""
        self.current_object_idx = (self.current_object_idx + 1) % len(self.objects)
    
    def get_current_object_name(self):
        """Get name of current object"""
        return self.objects[self.current_object_idx].name
