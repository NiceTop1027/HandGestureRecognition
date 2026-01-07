# Hand Gesture Recognition AR - Ultimate Edition 🦾

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange?style=for-the-badge&logo=google)

Advanced Hand Gesture Recognition system with **Dual Hand Control**, **Physics-based Interaction**, and **Futuristic HUD UI**. Experience an "Iron Man" like interface on your Mac.

## ✨ Key Features

### 1. 👐 Dual Hand Control (양손 분리 컨트롤)
*   **🟢 Left Hand (Rotation)**:
    *   **Grab & Spin**: Swipe to rotate the object with inertia.
    *   **Pinch Lock**: Physically grab the object to rotate it 1:1 with your hand.
*   **🔵 Right Hand (Scale)**:
    *   **Pinch Scale**: Adjust the size of the 3D object by pinching your thumb and index finger.

### 2. 🌪️ Natural Physics (물리 엔진)
*   **Inertial Spin**: Objects continue to spin after being thrown, simulating real-world momentum.
*   **Direct Manipulation**: "Grab" the object (Pinch) to stop rotation instanty and control it precisely.
*   **Weighted Feel**: Smoothed rotation algorithms provide a heavy, premium feel.

### 3. 🛡️ Premium HUD Interface
*   **Iron Man Style UI**: Rotating ring gauges around each hand.
*   **Energy Beams**: Visual connection lines between your hands and the 3D object.
*   **Real-time Feedback**: Color-coded indicators for Rotation (Green) and Scale (Blue).
*   **Glassmorphism**: High-performance blurred UI panels.

### 4. ⚡ Extreme Performance
*   **M-Series Optimization**: Optimized for Apple Silicon (M1/M2/M3/M4).
*   **120 FPS+**: ROI-based rendering and zero-latency loops for maximum fluidity.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/NiceTop1027/HandGestureRecognition.git
    cd HandGestureRecognition
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python hand_gesture_ar.py
    ```

## 🎮 How to Use

| Hand | Gesture | Action |
|:---:|---|---|
| **Left** | **Pinch (👌)** | **Grab & Rotate**: Lock rotation to hand movement. |
| **Left** | **Release (🖐)** | **Throw**: Release while moving to spin the object. |
| **Left** | **Fist (✊)** | **Stop**: Emergency brake for rotation. |
| **Right** | **Pinch (👌)** | **Scale**: Move fingers apart/together to resize. |
| **Right** | **Point (☝)** | **Move/Follow**: (Optional) Object follows finger. |

## ⚙️ Requirements

*   Python 3.11+
*   Webcam
*   MacOS (Recommended for Metal/M-chip optimization) or Windows/Linux

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
