"""
Visual effects engine for hand gesture AR
"""
import cv2
import numpy as np
import random
import math
from utils import get_hsv_color, lerp


class Particle:
    """Single particle for particle effects"""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-8, 8)
        self.vy = random.uniform(-12, -4)
        self.color = color
        self.life = 1.0
        self.size = random.randint(3, 10)
        self.gravity = 0.3
        
    def update(self):
        """Update particle physics"""
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.02
        return self.life > 0
    
    def draw(self, frame):
        """Draw particle on frame"""
        if self.life > 0:
            alpha = self.life
            size = int(self.size * self.life)
            color = tuple(int(c * alpha) for c in self.color)
            cv2.circle(frame, (int(self.x), int(self.y)), size, color, -1)


class ParticleSystem:
    """Manages particle effects"""
    def __init__(self):
        self.particles = []
        self.max_particles = 500
        
    def emit(self, x, y, count=30, color_hue=None):
        """
        Emit particles from a position
        
        Args:
            x, y: Emission position
            count: Number of particles to emit
            color_hue: HSV hue for particle color (random if None)
        """
        for _ in range(count):
            if len(self.particles) < self.max_particles:
                if color_hue is None:
                    hue = random.randint(0, 179)
                else:
                    hue = color_hue
                color = get_hsv_color(hue)
                self.particles.append(Particle(x, y, color))
    
    def update(self):
        """Update all particles"""
        self.particles = [p for p in self.particles if p.update()]
    
    def draw(self, frame):
        """Draw all particles"""
        for particle in self.particles:
            particle.draw(frame)
    
    def clear(self):
        """Clear all particles"""
        self.particles.clear()


class HologramEffect:
    """3D hologram effect above palm"""
    def __init__(self):
        self.rotation = 0
        self.visible = False
        self.position = (0, 0)
        self.size = 80
        
    def show(self, x, y):
        """Show hologram at position"""
        self.visible = True
        self.position = (x, y)
        
    def hide(self):
        """Hide hologram"""
        self.visible = False
        
    def update(self):
        """Update hologram animation"""
        self.rotation += 2
        if self.rotation >= 360:
            self.rotation = 0
    
    def draw(self, frame):
        """Draw hologram effect"""
        if not self.visible:
            return
        
        x, y = self.position
        size = self.size
        
        # Calculate 3D cube vertices
        angle_rad = math.radians(self.rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 3D cube vertices
        vertices_3d = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
        ]
        
        # Rotate and project to 2D
        vertices_2d = []
        for vx, vy, vz in vertices_3d:
            # Rotate around Y axis
            rotated_x = vx * cos_a + vz * sin_a
            rotated_z = -vx * sin_a + vz * cos_a
            
            # Simple orthographic projection
            scale = size / (2.5 - rotated_z * 0.3)
            px = int(x + rotated_x * scale)
            py = int(y + vy * scale * 0.7 - size // 2)  # Offset above palm
            vertices_2d.append((px, py))
        
        # Draw cube edges with neon glow effect
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Cyan hologram color
        color = (255, 255, 0)  # Cyan in BGR
        glow_color = (200, 200, 0)
        
        # Draw glow
        for start_idx, end_idx in edges:
            cv2.line(frame, vertices_2d[start_idx], vertices_2d[end_idx], glow_color, 4)
        
        # Draw main lines
        for start_idx, end_idx in edges:
            cv2.line(frame, vertices_2d[start_idx], vertices_2d[end_idx], color, 2)


class MagicTrail:
    """Magic trail effect following fingertips"""
    def __init__(self, max_trail_length=20):
        self.trails = [[] for _ in range(5)]  # One trail per finger
        self.max_length = max_trail_length
        self.hue_offset = 0
        
    def update(self, fingertip_positions, extended_fingers):
        """
        Update trails with new fingertip positions
        
        Args:
            fingertip_positions: List of (x, y) tuples for all 5 fingertips
            extended_fingers: List of indices of extended fingers
        """
        for i, pos in enumerate(fingertip_positions):
            if i in extended_fingers:
                # Add new position to trail
                self.trails[i].append(pos)
                if len(self.trails[i]) > self.max_length:
                    self.trails[i].pop(0)
            else:
                # Fade out trail for non-extended fingers
                if len(self.trails[i]) > 0:
                    self.trails[i].pop(0)
        
        # Update hue for rainbow effect
        self.hue_offset = (self.hue_offset + 1) % 180
    
    def draw(self, frame):
        """Draw all trails"""
        for finger_idx, trail in enumerate(self.trails):
            if len(trail) < 2:
                continue
            
            # Draw trail with thickness and color gradient
            for i in range(len(trail) - 1):
                start = trail[i]
                end = trail[i + 1]
                
                # Calculate properties based on position in trail
                alpha = (i + 1) / len(trail)
                thickness = int(lerp(2, 8, alpha))
                
                # Rainbow color based on finger and position
                hue = (self.hue_offset + finger_idx * 36 + i * 5) % 180
                color = get_hsv_color(hue)
                
                cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
    
    def clear(self):
        """Clear all trails"""
        self.trails = [[] for _ in range(5)]


class EffectsEngine:
    """Main effects engine managing all visual effects"""
    def __init__(self):
        self.particle_system = ParticleSystem()
        self.hologram = HologramEffect()
        self.magic_trail = MagicTrail()
        self.current_mode = 'particles'  # 'particles', 'hologram', 'trails', 'all'
        
    def handle_gesture(self, gesture, fingertip_positions, palm_center):
        """
        Handle gesture and trigger appropriate effects
        
        Args:
            gesture: Gesture dict from GestureDetector
            fingertip_positions: List of fingertip positions
            palm_center: Palm center position
        """
        gesture_type = gesture['type']
        changed = gesture['changed']
        extended = gesture['extended_fingers']
        
        # Open hand: Particle explosion
        if gesture_type == 'open_hand' and changed:
            for pos in fingertip_positions:
                self.particle_system.emit(pos[0], pos[1], count=20)
        
        # Fist: Clear all effects
        if gesture_type == 'fist' and changed:
            self.particle_system.clear()
            self.hologram.hide()
            self.magic_trail.clear()
        
        # Peace sign: Show hologram
        if gesture_type == 'peace':
            self.hologram.show(palm_center[0], palm_center[1])
        else:
            self.hologram.hide()
        
        # Pointing or any extended fingers: Magic trails
        if len(extended) > 0:
            self.magic_trail.update(fingertip_positions, extended)
        
    def update(self):
        """Update all effects"""
        self.particle_system.update()
        self.hologram.update()
    
    def draw(self, frame):
        """Draw all effects on frame"""
        self.particle_system.draw(frame)
        self.hologram.draw(frame)
        self.magic_trail.draw(frame)
