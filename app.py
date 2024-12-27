from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import torch
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = 'uploads/'
RESULTS_FOLDER = 'results/'
FRAMES_FOLDER = 'frames/'
MODEL_PATH = 'models/yolov5s.pt'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = torch.load(MODEL_PATH)
model.eval()
def extract_frames(video_path, output_folder):
    """Extract frames from the uploaded video."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    return count

def annotate_video(input_video, detections, output_video, shoplifting_percentage):
    """Annotate video with detection results and shoplifting percentage."""
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in detections:
            for (x1, y1, x2, y2, label, thief_likelihood) in detections[frame_idx]:
                # Check if person is detected as a thief
                if thief_likelihood > 70:  # Threshold for thief detection
                    color = (0, 0, 255)  # Red for thief
                    label = f"Thief {thief_likelihood:.2f}%"  # Show thief percentage
                else:
                    color = (0, 255, 0)  # Green for normal person detection
                
                # Draw bounding box with appropriate color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Annotate with overall shoplifting likelihood
        cv2.putText(frame, f"Shoplifting Likelihood: {shoplifting_percentage:.2f}%", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def is_theft_detected(x1, y1, x2, y2, movement_data):
    """Function to detect theft behavior (e.g., carrying large object or near an exit)."""
    object_width = x2 - x1
    object_height = y2 - y1
    
    # Placeholder logic: If bounding box is large, assume theft (carrying an object)
    if object_width > 100 and object_height > 100:  # Adjust these thresholds as needed
        return True  # Suspected theft

    # Dynamic analysis: check if a person is moving too fast or near exits
    speed_factor = movement_data.get('speed', 0)  # Assume speed factor is calculated
    if speed_factor > 1.5:  # High speed might indicate suspicious movement
        return True  # Speed suggests theft behavior

    return False

def calculate_thief_likelihood(x1, y1, x2, y2, previous_position=None, current_position=None):
    """Calculate the likelihood of being a thief based on dynamic factors."""
    width = x2 - x1
    height = y2 - y1

    # Example instinctive factor 1: larger bounding boxes (suspect larger objects are being carried)
    size_factor = (width * height) / 10000  # Normalize size factor (adjust denominator as needed)

    # Dynamic movement: if a person is moving quickly, it can increase thief likelihood
    movement_factor = 0
    if previous_position and current_position:
        distance_moved = np.sqrt((current_position[0] - previous_position[0])**2 +
                                 (current_position[1] - previous_position[1])**2)
        movement_factor = distance_moved / 5.0  # Adjust divisor for movement scaling

    # Combined instinctive likelihood
    thief_likelihood = (size_factor * 100 + movement_factor * 100) / 2  # Average of size and movement factors
    thief_likelihood = min(thief_likelihood, 100)  # Cap at 100%

    # Return the likelihood only if it's above 70%
    return thief_likelihood if thief_likelihood > 70 else 0  # Only return likelihood if > 70%

def detect_objects_in_frames(frames_folder):
    """Run object detection on all frames and track suspicious behavior."""
    detections = {}
    suspicious_frames = 0
    total_frames = 0
    previous_positions = {}

    for frame_file in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_file)
        results = model(frame_path)
        frame_idx = int(frame_file.split('_')[1].split('.')[0])
        detections[frame_idx] = []
        
        for box in results.xyxy[0].numpy():
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = results.names[int(box[5])]
            
            thief_likelihood = 0
            previous_position = previous_positions.get(frame_idx - 1, {}).get(label, None)
            current_position = (x1, y1)

            if label == 'person' and is_theft_detected(x1, y1, x2, y2, movement_data={}):
                thief_likelihood = calculate_thief_likelihood(x1, y1, x2, y2, previous_position, current_position)

            detections[frame_idx].append((x1, y1, x2, y2, label, thief_likelihood))

            # Track suspicious behavior (e.g., person near exit, carrying large items)
            if label == 'person' and thief_likelihood > 0:  # Only count if thief likelihood > 0
                suspicious_frames += 1
            previous_positions[frame_idx] = {label: (x1, y1)}  # Store current position

        total_frames += 1

    # Calculate percentage of suspicious frames
    shoplifting_percentage = (suspicious_frames / total_frames) * 100 if total_frames > 0 else 0
    return detections, shoplifting_percentage

@app.route('/')
def home():
    """Render the upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing."""
    if 'video' not in request.files:
        return "No video file uploaded", 400

    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # Extract frames
    frame_count = extract_frames(video_path, FRAMES_FOLDER)

    # Detect objects in frames and calculate shoplifting percentage
    detections, shoplifting_percentage = detect_objects_in_frames(FRAMES_FOLDER)

    # Annotate video with detections and shoplifting percentage
    output_video_path = os.path.join(RESULTS_FOLDER, f"annotated_{video.filename}")
    annotate_video(video_path, detections, output_video_path, shoplifting_percentage)

    return f"Video processed successfully! <a href='/results/{os.path.basename(output_video_path)}'>Download Results</a>"

@app.route('/results/<filename>')
def download_results(filename):
    """Serve the processed video."""
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
