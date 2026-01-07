"""
Particle System for Hand Gesture Effects
"""
import cv2
import numpy as np
import random


class Particle:
    """Single particle"""
    def __init__(self, x, y, color, velocity=None):
        self.x = x
        self.y = y
        self.vx = velocity[0] if velocity else random.uniform(-5, 5)
        self.vy = velocity[1] if velocity else random.uniform(-8, -2)
        self.color = color
        self.life = 1.0
        self.size = random.randint(2, 6)
        self.gravity = 0.2
        
    def update(self):
        """Update particle physics"""
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.015
        return self.life > 0
    
    def draw(self, frame):
        """Draw particle"""
        if self.life > 0:
            alpha = self.life
            size = max(1, int(self.size * self.life))
            
            # Glow effect
            glow_size = size + 2
            glow_color = tuple(int(c * alpha * 0.5) for c in self.color)
            cv2.circle(frame, (int(self.x), int(self.y)), glow_size, glow_color, -1, cv2.LINE_AA)
            
            # Main particle
            color = tuple(int(c * alpha) for c in self.color)
            cv2.circle(frame, (int(self.x), int(self.y)), size, color, -1, cv2.LINE_AA)


class ParticleSystem:
    """Manages particle effects"""
    
    def __init__(self, max_particles=300):
        self.particles = []
        self.max_particles = max_particles
        
        # Color palette
        self.colors = [
            (147, 112, 219),  # Purple
            (100, 200, 255),  # Cyan
            (80, 220, 120),   # Green
            (255, 150, 100),  # Orange
            (255, 100, 200),  # Pink
        ]
    
    def emit(self, x, y, count=15, color_idx=None, spread=1.0):
        """Emit particles"""
        if len(self.particles) >= self.max_particles:
            return
        
        for _ in range(min(count, self.max_particles - len(self.particles))):
            if color_idx is None:
                color = random.choice(self.colors)
            else:
                color = self.colors[color_idx % len(self.colors)]
            
            # Random velocity with spread
            vx = random.uniform(-8, 8) * spread
            vy = random.uniform(-12, -4) * spread
            
            self.particles.append(Particle(x, y, color, velocity=(vx, vy)))
    
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
    
    def get_count(self):
        """Get current particle count"""
        return len(self.particles)
