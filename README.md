# 🛍️ Shoplifting Detection System

An AI-powered surveillance solution for detecting shoplifting incidents in real-time using deep learning. This system leverages a hybrid neural network (CNN + GRU) to analyze video feeds and classify behavior as normal or suspicious, enhancing security in retail environments.

---

## 🚀 Features

- Real-time video surveillance and prediction
- CNN for spatial feature extraction from video frames
- GRU for analyzing temporal patterns in behavior
- High accuracy: 93% on UCF-Crime dataset
- Web interface built with Flask for user interaction
- Visualization of predictions on video frames

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **Deep Learning:** TensorFlow, Keras, OpenCV  
- **Model:** Convolutional Neural Network (CNN) + Gated Recurrent Unit (GRU)  
- **Frontend:** HTML, Bootstrap (for Flask interface)  

---

## 📂 Dataset

- **Name:** [UCF-Crime Dataset](http://crcv.ucf.edu/projects/real-world/)
- **Usage:** Extracted video segments labeled for 'shoplifting' and 'normal' behavior.
- **Preprocessing:** Frame extraction, resizing, normalization, and sequence batching.

---

## ▶️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/shoplifting-detection.git
   cd shoplifting-detection
