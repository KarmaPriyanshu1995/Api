# from flask import Flask, request, send_file, jsonify
# import cv2
# import numpy as np
# import io
# import os
# from moviepy.editor import ImageSequenceClip
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "output"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# def remove_bg(image):
#     mask = np.zeros(image.shape[:2], np.uint8)
#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)
#     height, width = image.shape[:2]
#     rect = (10, 10, width-10, height-10)
#     cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#     result = image * mask2[:, :, np.newaxis]
#     return result

# @app.route('/remove-bg', methods=['POST'])
# def remove_bg_api():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file provided'}), 400

#         uploaded_file = request.files['image']
#         in_memory_file = io.BytesIO(uploaded_file.read())
#         image = cv2.imdecode(np.frombuffer(in_memory_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

#         output_image = remove_bg(image)

#         _, buffer = cv2.imencode('.png', output_image)
#         output_stream = io.BytesIO(buffer)

#         return send_file(output_stream, mimetype='image/png')
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# def remove_bg_from_frame(frame):
#     return remove_bg(frame)

# def process_video(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_no_bg = remove_bg_from_frame(frame)
#         frames.append(frame_no_bg)

#     cap.release()
#     clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
#     clip.write_videofile(output_path, codec="libx264")

# @app.route("/remove-bg-video", methods=["POST"])
# def remove_bg_video():
#     try:
#         if "video" not in request.files:
#             return jsonify({"error": "No video file provided"}), 400

#         video = request.files["video"]
#         input_path = os.path.join(UPLOAD_FOLDER, video.filename)
#         output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")

#         video.save(input_path)
#         process_video(input_path, output_path)

#         return send_file(output_path, as_attachment=True)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=8000)
from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to remove background using GrabCut
def remove_bg(image):
    try:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        height, width = image.shape[:2]
        rect = (10, 10, width-10, height-10)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        result = image * mask2[:, :, np.newaxis]
        return result
    except Exception as e:
        print(f"Error removing background: {e}")
        return None

# API to remove background from an image
@app.route("/remove-bg", methods=["POST"])
def remove_bg_api():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        uploaded_file = request.files["image"]
        in_memory_file = io.BytesIO(uploaded_file.read())
        image = cv2.imdecode(np.frombuffer(in_memory_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        output_image = remove_bg(image)

        if output_image is None:
            return jsonify({"error": "Failed to process image"}), 500

        _, buffer = cv2.imencode(".png", output_image)
        output_stream = io.BytesIO(buffer)

        return send_file(output_stream, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Process a video frame-by-frame
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Video codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no_bg = remove_bg(frame)
        if frame_no_bg is not None:
            out.write(frame_no_bg)

    cap.release()
    out.release()

# API to remove background from a video
@app.route("/remove-bg-video", methods=["POST"])
def remove_bg_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video = request.files["video"]
        filename = secure_filename(video.filename)

        if filename == "":
            return jsonify({"error": "Invalid file name"}), 400

        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")

        video.save(input_path)  # Save uploaded video file
        process_video(input_path, output_path)  # Process video

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port=8000)
