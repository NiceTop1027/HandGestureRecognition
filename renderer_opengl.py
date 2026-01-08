import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import math

class RendererGL:
    def __init__(self, width, height):
        self.display_width = width
        self.display_height = height
        
        # Initialize Pygame and OpenGL
        pygame.init()
        pygame.display.set_caption("Premium Hand Gesture AR (Holographic Edition)")
        
        # Multisampling for smoother edges
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        
        # OpenGL Configuration
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Camera Texture (Background)
        textures = glGenTextures(2)
        if isinstance(textures, (list, tuple, np.ndarray)):
            self.cam_tex_id = textures[0]
            self.ui_tex_id = textures[1]
        else:
            self.cam_tex_id = textures
            self.ui_tex_id = glGenTextures(1)
            
        print(f"DEBUG: Texture IDs generated: Cam={self.cam_tex_id}, UI={self.ui_tex_id}")
        
        # Lighting - "Sci-Fi" setup
        # Blue-ish ambient light
        glLightfv(GL_LIGHT0, GL_POSITION,  (5, 10, 10, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.1, 0.1, 0.2, 1.0)) 
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.9, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        
        # Camera Setup
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (width / height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        self.grid_rotation = 0
        
    def update_texture(self, texture_id, image):
        """Update OpenGL texture from OpenCV image"""
        if image is None: return
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 0) # Flip vertically for OpenGL
        
        img_data = image.tobytes()
        width, height = image.shape[1], image.shape[0]
        
        try:
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        except Exception as e:
            print(f"Error updating texture {texture_id}: {e}")

    def draw_background(self):
        """Draw webcam feed as full-screen background quad"""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.cam_tex_id)
        
        glColor3f(1.0, 1.0, 1.0)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display_width, 0, self.display_height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(self.display_width, 0)
        glTexCoord2f(1, 1); glVertex2f(self.display_width, self.display_height)
        glTexCoord2f(0, 1); glVertex2f(0, self.display_height)
        glEnd()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_BLEND)

    def draw_grid(self):
        """Draw a sci-fi rotating grid floor"""
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive blending for glow
        
        glPushMatrix()
        glTranslatef(0, -2, 0)
        glRotatef(self.grid_rotation, 0, 1, 0)
        self.grid_rotation = (self.grid_rotation + 0.2) % 360
        
        glBegin(GL_LINES)
        size = 10
        steps = 20
        start = -size
        end = size
        step_val = (end - start) / steps
        
        for i in range(steps + 1):
            val = start + i * step_val
            
            # Fade out at edges
            alpha = (1.0 - abs(val)/size) * 0.5
            
            # Cyan grid
            glColor4f(0.0, 0.8, 1.0, alpha)
            
            glVertex3f(val, 0, start)
            glVertex3f(val, 0, end)
            
            glVertex3f(start, 0, val)
            glVertex3f(end, 0, val)
        glEnd()
        
        # Draw a central ring
        glBegin(GL_LINE_LOOP)
        for i in range(36):
            angle = i * (2 * math.pi / 36)
            x = math.cos(angle) * 3
            z = math.sin(angle) * 3
            glColor4f(1.0, 0.0, 1.0, 0.3) # Magenta ring
            glVertex3f(x, 0, z)
        glEnd()
        
        glPopMatrix()
        glEnable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw_cube(self):
        """Holographic Cube"""
        glDisable(GL_TEXTURE_2D)
        
        # 1. Inner Translucent Body
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE) # Don't write to depth buffer for transparency
        
        glBegin(GL_QUADS)
        colors = [
            (0.0, 1.0, 1.0, 0.2), # Cyan
            (0.0, 0.8, 1.0, 0.2),
            (0.0, 0.6, 1.0, 0.2),
            (0.0, 0.4, 1.0, 0.2),
            (0.0, 0.8, 1.0, 0.2),
            (0.0, 0.8, 1.0, 0.2)
        ]
        
        # Faces
        faces = [
            # Front
            ([(-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)], (0,0,1)),
            # Back
            ([(-1,-1,-1), (-1,1,-1), (1,1,-1), (1,-1,-1)], (0,0,-1)),
            # Top
            ([(-1,1,-1), (-1,1,1), (1,1,1), (1,1,-1)], (0,1,0)),
            # Bottom
            ([(-1,-1,-1), (1,-1,-1), (1,-1,1), (-1,-1,1)], (0,-1,0)),
            # Right
            ([(1,-1,-1), (1,1,-1), (1,1,1), (1,-1,1)], (1,0,0)),
            # Left
            ([(-1,-1,-1), (-1,-1,1), (-1,1,1), (-1,1,-1)], (-1,0,0))
        ]
        
        for i, (verts, normal) in enumerate(faces):
            glNormal3f(*normal)
            glColor4f(*colors[i])
            for v in verts:
                glVertex3f(*v)
        glEnd()
        
        glDepthMask(GL_TRUE)
        
        # 2. Wireframe Overlay (Additive)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        glLineWidth(2.0)
        glColor4f(1.0, 1.0, 1.0, 0.6) # White glow
        
        glBegin(GL_LINES)
        edges = [
            ((-1,-1,1), (1,-1,1)), ((1,-1,1), (1,1,1)), ((1,1,1), (-1,1,1)), ((-1,1,1), (-1,-1,-1)),
            ((-1,-1,-1), (-1,1,-1)), ((-1,1,-1), (1,1,-1)), ((1,1,-1), (1,-1,-1)), ((1,-1,-1), (-1,-1,-1)),
            ((-1,-1,1), (-1,-1,-1)), ((1,-1,1), (1,-1,-1)), ((1,1,1), (1,1,-1)), ((-1,1,1), (-1,1,-1)),
            ((-1,-1,1), (-1,1,1)) # Fix first edge loop
        ]
        
        # Re-define exact edges for cube
        cube_edges = [
            # Front face
            (-1,-1,1), (1,-1,1),  (1,-1,1), (1,1,1),  (1,1,1), (-1,1,1),  (-1,1,1), (-1,-1,1),
            # Back face
            (-1,-1,-1), (-1,1,-1), (-1,1,-1), (1,1,-1), (1,1,-1), (1,-1,-1), (1,-1,-1), (-1,-1,-1),
            # Connecting
            (-1,-1,1), (-1,-1,-1), (1,-1,1), (1,-1,-1), (1,1,1), (1,1,-1), (-1,1,1), (-1,1,-1)
        ]
        
        for i in range(0, len(cube_edges), 2):
            glVertex3f(*cube_edges[i])
            glVertex3f(*cube_edges[i+1])
            
        glEnd()
        
        glEnable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw_sphere(self):
        """Holographic Sphere"""
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        
        # Transparent body
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        
        glColor4f(0.0, 0.8, 1.0, 0.3)
        gluSphere(quad, 1.0, 32, 32)
        
        glDepthMask(GL_TRUE)
        
        # Wireframe overlay
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        glColor4f(1.0, 1.0, 1.0, 0.4)
        gluQuadricDrawStyle(quad, GLU_LINE)
        gluSphere(quad, 1.01, 16, 16) # Slightly larger
        gluQuadricDrawStyle(quad, GLU_FILL)
        
        glEnable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        gluDeleteQuadric(quad)
            
    def draw_pyramid(self):
        """Holographic Pyramid"""
        # Similar logic: Translucent body + Wireframe
        glDisable(GL_TEXTURE_2D)
        
        verts = [
            (0, 1.5, 0), # Top
            (-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1) # Base
        ]
        
        # Body
        glDepthMask(GL_FALSE) 
        glEnable(GL_BLEND)
        
        glBegin(GL_TRIANGLES)
        colors = [(1,0,0,0.3), (0,1,0,0.3), (0,0,1,0.3), (1,1,0,0.3)]
        
        # Sides
        indices = [(0,1,2), (0,2,3), (0,3,4), (0,4,1)]
        for i, idxs in enumerate(indices):
            glColor4f(*colors[i])
            for idx in idxs:
                glVertex3f(*verts[idx])
        glEnd()
        
        # Base
        glBegin(GL_QUADS)
        glColor4f(1,0,1,0.3)
        for i in [1,2,3,4]:
            glVertex3f(*verts[i])
        glEnd()
        
        glDepthMask(GL_TRUE)
        
        # Wireframe
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glLineWidth(2.0)
        glColor4f(1.0, 1.0, 1.0, 0.6)
        
        glBegin(GL_LINES)
        segments = [
            (0,1), (0,2), (0,3), (0,4), # Sides
            (1,2), (2,3), (3,4), (4,1)  # Base
        ]
        for s, e in segments:
            glVertex3f(*verts[s])
            glVertex3f(*verts[e])
        glEnd()
        
        glEnable(GL_LIGHTING)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def render_scene(self, cam_frame, rotation_angles, scale, ui_overlay=None, shape_type="Cube"):
        """
        Main Render Function
        """
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # 1. Background (Camera Feed)
        if cam_frame is not None:
            self.update_texture(self.cam_tex_id, cam_frame)
            self.draw_background()
            
        # 2. 3D Scene
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -6.0) # Camera distance
        
        # Draw floor grid
        self.draw_grid()
        
        # Transform Object
        glPushMatrix()
        
        # Rotation
        glRotatef(rotation_angles[0] * 57.29, 1, 0, 0) # Pitch
        glRotatef(rotation_angles[1] * 57.29, 0, 1, 0) # Yaw
        glRotatef(rotation_angles[2] * 57.29, 0, 0, 1) # Roll
        
        # Scale
        s = scale / 100.0 * 0.8
        glScalef(s, s, s)
        
        # Draw Main Shape
        if shape_type == "Sphere":
            self.draw_sphere()
        elif shape_type == "Pyramid":
            self.draw_pyramid()
        else:
            self.draw_cube()
            
        glPopMatrix()
            
        # 3. UI Overlay
        if ui_overlay is not None:
             glDisable(GL_DEPTH_TEST)
             glDisable(GL_LIGHTING)
             glEnable(GL_BLEND)
             glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
             
             self.update_texture(self.ui_tex_id, ui_overlay)
             glBindTexture(GL_TEXTURE_2D, self.ui_tex_id)
             glEnable(GL_TEXTURE_2D)
             
             glColor4f(1, 1, 1, 1) # Full opacity
             
             glMatrixMode(GL_PROJECTION)
             glPushMatrix()
             glLoadIdentity()
             glOrtho(0, self.display_width, 0, self.display_height, -1, 1)
             glMatrixMode(GL_MODELVIEW)
             glPushMatrix()
             glLoadIdentity()
             
             glBegin(GL_QUADS)
             glTexCoord2f(0, 0); glVertex2f(0, 0)
             glTexCoord2f(1, 0); glVertex2f(self.display_width, 0)
             glTexCoord2f(1, 1); glVertex2f(self.display_width, self.display_height)
             glTexCoord2f(0, 1); glVertex2f(0, self.display_height)
             glEnd()
             
             glPopMatrix()
             glMatrixMode(GL_PROJECTION)
             glPopMatrix()
             glMatrixMode(GL_MODELVIEW)
             
             glEnable(GL_DEPTH_TEST)
             glEnable(GL_LIGHTING)

        return pygame.key.get_pressed()
        
    def flip(self):
        pygame.display.flip()
        
    def quit(self):
        pygame.quit()
