from flask import Flask, request, jsonify, render_template, Response
import os
import cv2
from ultralytics import YOLO    # Import YOLO from the ultralytics package
import glob

app = Flask(__name__)

# Define directories for uploading files and saving results
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Folder to store uploaded files
RESULT_FOLDER = os.path.join(os.getcwd(), 'static', 'results')  # Folder to store result files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your trained YOLO model manually
model = YOLO('helmet_detection_model.pt')  # Update with the correct path to your model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the uploaded image
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        
        # Run detection on the image
        results = model(input_path)
        result = results[0]  # Assuming there's at least one detection
        
        # Save the result image
        result_image_path = os.path.join(RESULT_FOLDER, file.filename)
        result.save(result_image_path)
        
        # Construct the result URL
        result_image_url = f"/static/results/{file.filename}"
        
        return jsonify({'result_image': result_image_url})
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/upload-video', methods=['POST'])
def upload_video():
    try:
        # Step 1: Check for 'video' in the request
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Step 2: Save the uploaded video
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        print(f"Uploaded video saved to {input_path}")

        # Step 3: Run YOLO detection on the video
        print("Running YOLO detection on the video")
        results = model(input_path, save=True)  # YOLO saves output in runs/detect/predict*

        # Step 4: Dynamically find the latest YOLO output directory
        yolo_output_dirs = glob.glob(os.path.join(os.getcwd(), 'runs', 'detect', 'predict*'))
        if not yolo_output_dirs:
            print("Error: No YOLO output directory found")
            return jsonify({'error': 'YOLO output directory not found'}), 500

        latest_output_dir = max(yolo_output_dirs, key=os.path.getctime)  # Get the most recently modified directory
        print(f"Latest YOLO output directory: {latest_output_dir}")

        # Step 5: Find the processed video in the latest output directory
        processed_video_path = os.path.join(latest_output_dir, file.filename)
        if not os.path.exists(processed_video_path):
            print(f"Error: Processed video not found at {processed_video_path}")
            return jsonify({'error': 'Processed video not found'}), 500

        # Step 6: Move the processed video to RESULT_FOLDER
        result_video_path = os.path.join(RESULT_FOLDER, file.filename)
        os.rename(processed_video_path, result_video_path)
        print(f"Processed video moved to: {result_video_path}")

        # Step 7: Construct the result URL
        result_video_url = f"/static/results/{file.filename}"
        print(f"Processed video URL: {result_video_url}")

        return jsonify({'result_video': result_video_url})
    
    

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500



# Initialize the camera globally (to avoid opening multiple cameras)
cap = None

@app.route('/video_feed')
def video_feed():
    global cap
    # Open the webcam only if it's not already opened
    if cap is None:
        cap = cv2.VideoCapture(0)  # Open default camera
        if not cap.isOpened():
            return jsonify({'error': 'Could not access the webcam'}), 500

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no frame is captured

            # Perform detection
            results = model(frame)
            result_frame = results[0].plot()  # Get the frame with detection results

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', result_frame)
            if not ret:
                break  # If encoding fails, break the loop

            # Convert the frame to bytes for streaming
            frame_bytes = buffer.tobytes()

            # Yield the frame as part of the MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    # Response as streaming content
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed', methods=['POST'])
def stop_video_feed():
    global cap
    if cap:
        cap.release()  # Release the webcam when stopping the feed
        cap = None  # Reset the camera object to None
    return jsonify({'message': 'Camera feed stopped'}), 200

if __name__ == '__main__':
    app.run(debug=True)
