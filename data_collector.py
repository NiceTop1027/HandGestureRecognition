"""
Data Collector for Hand Gesture Recognition
Captures hand landmarks and saves them to a CSV file for training.
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

class DataCollector:
    def __init__(self, output_file='gesture_data.csv'):
        self.output_file = output_file
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.output_file):
            columns = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.output_file, index=False)
            
    def process_landmarks(self, landmarks):
        """Convert landmarks to a flat list of normalized coordinates"""
        # Normalize relative to wrist (landmark 0) to make it position-invariant
        wrist = landmarks[0]
        base_x, base_y, base_z = wrist.x, wrist.y, wrist.z
        
        coords = []
        for lm in landmarks:
            # Relative coordinates
            coords.append(lm.x - base_x)
            coords.append(lm.y - base_y)
            coords.append(lm.z - base_z)
            
        # Alternative: Just return raw coordinates if we augment data later
        # But relative coordinates are usually better for simple classifiers
        
        # Let's try raw normalized coordinates first + flatten
        # flattend structure: x0, x1, ..., y0, y1, ... z0, z1...
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        zs = [lm.z for lm in landmarks]
        
        return xs + ys + zs

    def collect(self):
        cap = cv2.VideoCapture(0)
        print("📷 Data Collector Started")
        print("Press 'A'-'Z' to save a sample for that letter.")
        print("Press '0'-'9' for numbers.")
        print("Press ESC to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Store current landmarks
                    self.current_landmarks = hand_landmarks.landmark
            
            cv2.imshow('Data Collector', frame)
            
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            elif key != -1:
                char = chr(key).upper()
                if (char >= 'A' and char <= 'Z') or (char >= '0' and char <= '9'):
                    if hasattr(self, 'current_landmarks'):
                        self.save_sample(char, self.current_landmarks)
                        print(f"✅ Saved sample for '{char}'")
                        
                        # Visual feedback
                        cv2.putText(frame, f"Saved: {char}", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Data Collector', frame)
                        cv2.waitKey(200) # Short pause to show feedback
                    else:
                        print("⚠️ No hand detected!")

        cap.release()
        cv2.destroyAllWindows()

    def save_sample(self, label, landmarks):
        data = self.process_landmarks(landmarks)
        row = [label] + data
        
        df = pd.DataFrame([row])
        df.to_csv(self.output_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect()
