"""
Hand gesture detection and classification - Improved Version
"""
import math


class GestureDetector:
    """Detect hand gestures from MediaPipe landmarks with improved accuracy"""
    
    # Finger tip and base landmarks indices
    FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    FINGER_MIDS = [3, 7, 11, 15, 19]  # MCP joints (middle of finger)
    FINGER_PIPS = [2, 6, 10, 14, 18]  # PIP joints for each finger
    
    def __init__(self):
        self.previous_gesture = None
        self.gesture_history = []
        self.max_history = 3  # Reduced for faster response
        
    def get_distance(self, point1, point2):
        """Calculate distance between two landmark points"""
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )
    
    def is_finger_extended(self, landmarks, finger_index):
        """
        Robust finger extension detection (Rotation Invariant)
        Uses distance from wrist to determine if finger is extended.
        
        Args:
            landmarks: List of hand landmarks
            finger_index: Finger index (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
        
        Returns:
            bool: True if finger is extended
        """
        tip_idx = self.FINGER_TIPS[finger_index]
        pip_idx = self.FINGER_PIPS[finger_index]
        MCP_idx = self.FINGER_MIDS[finger_index] if hasattr(self, 'FINGER_MIDS') else (tip_idx - 2)
        wrist = landmarks[0]
        
        # Calculate distance from wrist to tip and wrist to PIP (or MCP)
        # Using squared distance is faster (no sqrt)
        def dist_sq(p1, p2):
            return (p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2

        d_tip_wrist = dist_sq(landmarks[tip_idx], wrist)
        d_pip_wrist = dist_sq(landmarks[pip_idx], wrist)
        
        # Special case for thumb
        if finger_index == 0:
            # Thumb: Check distance to Pinky MCP (Landmark 17)
            # When folded, thumb tip is close to pinky base.
            # When extended, it is far.
            
            pinky_mcp = landmarks[17]
            index_mcp = landmarks[5]
            
            d_tip_pinky = dist_sq(landmarks[tip_idx], pinky_mcp)
            d_mcp_pinky = dist_sq(landmarks[2], pinky_mcp)
            
            # Use palm width (Index MCP to Pinky MCP) as a scale reference
            d_palm_width = dist_sq(index_mcp, pinky_mcp)
            
            # Thumb is extended if its tip is farther from pinky base than its own MCP is
            # AND it is significantly far from the pinky base (greater than palm width roughly)
            return d_tip_pinky > d_palm_width and d_tip_pinky > d_mcp_pinky
        else:
            # For other fingers: Tip should be farther from wrist than PIP
            # This works regardless of hand rotation (up, down, sideways)
            return d_tip_wrist > d_pip_wrist * 1.2
    
    def count_extended_fingers(self, landmarks):
        """
        Count how many fingers are extended with improved accuracy
        
        Args:
            landmarks: List of hand landmarks
        
        Returns:
            tuple: (count, extended_fingers_list)
        """
        extended = []
        
        for i in range(5):
            if self.is_finger_extended(landmarks, i):
                extended.append(i)
        
        return len(extended), extended
    
    def detect_gesture(self, landmarks):
        """
        Detect gesture type based on extended fingers
        
        Args:
            landmarks: List of hand landmarks
        
        Returns:
            dict: Gesture information
        """
        if not landmarks:
            return {
                'type': 'none',
                'name': 'No Hand',
                'finger_count': 0,
                'extended_fingers': []
            }
        
        count, extended = self.count_extended_fingers(landmarks)
        
        # Classify gesture based on finger count and configuration
        gesture_type = 'unknown'
        gesture_name = 'Unknown'
        
        if count == 0:
            gesture_type = 'fist'
            gesture_name = 'Fist ✊'
        elif count == 1:
            if 1 in extended:  # Index finger
                gesture_type = 'pointing'
                gesture_name = 'Pointing ☝️'
            elif 0 in extended:  # Thumb only
                gesture_type = 'thumbs_up'
                gesture_name = 'Thumbs Up 👍'
            else:
                gesture_type = 'one'
                gesture_name = 'One Finger'
        elif count == 2:
            if 1 in extended and 2 in extended:  # Index and middle
                gesture_type = 'peace'
                gesture_name = 'Peace ✌️'
            elif 0 in extended and 1 in extended:  # Thumb and index
                gesture_type = 'two'
                gesture_name = 'Two Fingers'
            else:
                gesture_type = 'two'
                gesture_name = 'Two Fingers'
        elif count == 3:
            gesture_type = 'three'
            gesture_name = 'Three 🤟'
        elif count == 4:
            gesture_type = 'four'
            gesture_name = 'Four Fingers'
        elif count == 5:
            gesture_type = 'open_hand'
            gesture_name = 'Open Hand 🖐️'
        
        # Update history for stability
        self.gesture_history.append(gesture_type)
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
        
        # Check if gesture just changed
        gesture_changed = (self.previous_gesture != gesture_type)
        self.previous_gesture = gesture_type
        
        return {
            'type': gesture_type,
            'name': gesture_name,
            'finger_count': count,
            'extended_fingers': extended,
            'changed': gesture_changed,
            'stable': len(set(self.gesture_history)) == 1 if self.gesture_history else False
        }
    
    def get_fingertip_positions(self, landmarks, width, height):
        """
        Get pixel positions of all fingertips
        
        Args:
            landmarks: List of hand landmarks
            width: Frame width
            height: Frame height
        
        Returns:
            list: List of (x, y) tuples for each fingertip
        """
        positions = []
        for tip_idx in self.FINGER_TIPS:
            x = int(landmarks[tip_idx].x * width)
            y = int(landmarks[tip_idx].y * height)
            positions.append((x, y))
        return positions
    
    def get_palm_center(self, landmarks, width, height):
        """
        Calculate palm center position
        
        Args:
            landmarks: List of hand landmarks
            width: Frame width
            height: Frame height
        
        Returns:
            tuple: (x, y) palm center position
        """
        # Use landmarks 0, 5, 9, 13, 17 (wrist and finger bases)
        palm_landmarks = [0, 5, 9, 13, 17]
        x_sum = sum(landmarks[i].x for i in palm_landmarks)
        y_sum = sum(landmarks[i].y for i in palm_landmarks)
        
        x = int((x_sum / len(palm_landmarks)) * width)
        y = int((y_sum / len(palm_landmarks)) * height)
        
        return (x, y)
    
    def get_hand_orientation(self, landmarks):
        """
        Calculate hand orientation angles for 3D control
        """
        import math
        
        # Get key points
        wrist = landmarks[0]
        middle_base = landmarks[9]
        index_base = landmarks[5]
        pinky_base = landmarks[17]
        
        # Calculate pitch (up/down tilt)
        dy = middle_base.y - wrist.y
        dz = middle_base.z - wrist.z
        pitch = math.atan2(dy, abs(dz) + 0.1)
        
        # Calculate yaw (left/right rotation)
        dx = index_base.x - pinky_base.x
        dz = index_base.z - pinky_base.z
        yaw = math.atan2(dx, abs(dz) + 0.1) * 2
        
        # Calculate roll
        palm_dx = index_base.x - pinky_base.x
        palm_dy = index_base.y - pinky_base.y
        roll = math.atan2(palm_dy, palm_dx)
        
        return (pitch, yaw, roll)

    def get_pinch_distance(self, landmarks, width, height):
        """
        Calculate distance between thumb tip and index tip
        
        Returns:
            float: Normalized distance (0.0 to 1.0 approx) or None
        """
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate Euclidean distance
        dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        return dist
