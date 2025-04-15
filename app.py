from flask import Flask, request, render_template, send_from_directory, jsonify
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
MODEL_PATH = 'models/detection_model.pt'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Replace with custom model if needed

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
                if thief_likelihood > 70:
                    color = (0, 0, 255)  # Red for thief
                    label = f"Thief {thief_likelihood:.2f}%"
                else:
                    color = (0, 255, 0)  # Green for normal person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Shoplifting Likelihood: {shoplifting_percentage:.2f}%", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def is_theft_detected(x1, y1, x2, y2, movement_data):
    """Function to detect theft behavior."""
    object_width = x2 - x1
    object_height = y2 - y1
    if object_width > 100 and object_height > 100:
        return True
    speed_factor = movement_data.get('speed', 0)
    if speed_factor > 1.5:
        return True
    return False

def calculate_thief_likelihood(x1, y1, x2, y2, previous_position=None, current_position=None):
    """Calculate the likelihood of being a thief."""
    width = x2 - x1
    height = y2 - y1
    size_factor = (width * height) / 10000
    movement_factor = 0
    if previous_position and current_position:
        distance_moved = np.sqrt((current_position[0] - previous_position[0])**2 +
                               (current_position[1] - previous_position[1])**2)
        movement_factor = distance_moved / 5.0
    thief_likelihood = (size_factor * 100 + movement_factor * 100) / 2
    thief_likelihood = min(thief_likelihood, 100)
    return thief_likelihood if thief_likelihood > 70 else 0

def detect_objects_in_frames(frames_folder):
    """Run object detection on all frames."""
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

            if label == 'person' and thief_likelihood > 0:
                suspicious_frames += 1
            previous_positions[frame_idx] = {label: (x1, y1)}

        total_frames += 1

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
        return jsonify({'success': False, 'message': 'No video file uploaded'}), 400

    video = request.files['video']
    video_filename = video.filename
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video.save(video_path)

    try:
        # Extract frames
        frame_count = extract_frames(video_path, FRAMES_FOLDER)

        # Detect objects and calculate shoplifting percentage
        detections, shoplifting_percentage = detect_objects_in_frames(FRAMES_FOLDER)

        # Annotate video
        output_filename = f"annotated_{video_filename}"
        output_video_path = os.path.join(RESULTS_FOLDER, output_filename)
        annotate_video(video_path, detections, output_video_path, shoplifting_percentage)

        return jsonify({
            'success': True,
            'filename': output_filename,
            'message': 'Video processed successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing video: {str(e)}'
        }), 500

@app.route('/results/<filename>')
def download_results(filename):
    """Serve the processed video."""
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)