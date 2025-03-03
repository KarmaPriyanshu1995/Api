from flask import Flask, request, send_file, jsonify, Response, stream_with_context
import cv2
import numpy as np
import io
import os
import time
import mediapipe as mp
from werkzeug.utils import secure_filename
import logging
import requests
from PIL import Image
import torch
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create absolute paths to ensure correct file handling
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "output")
MODEL_FOLDER = os.path.join(CURRENT_DIR, "models")

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Initialize MediaPipe Selfie Segmentation for backup
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize U2Net model for high-quality segmentation
def initialize_u2net():
    try:
        # Check if the model file exists
        model_path = os.path.join(MODEL_FOLDER, "u2net.pth")
        if not os.path.exists(model_path):
            # Download the model if it doesn't exist
            logger.info("Downloading U2Net model...")
            download_model("u2net", model_path)
        
        # Import U2Net (only when needed to avoid import errors)
        from u2net_model import U2NET
        
        # Load the model
        model = U2NET(3, 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info("U2Net model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error initializing U2Net: {e}")
        return None

# Helper function to download U2Net model
def download_model(model_name, output_path):
    try:
        # U2Net model URL
        url = "https://github.com/xuebinqin/U-2-Net/releases/download/NeurIPS2020/u2net.pth"
        
        # Download the model
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logger.info(f"Model downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

# Load model only when needed (lazy loading)
u2net_model = None

# Function to normalize the image for U2Net
def normalize(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Perfect background removal using U2Net
def remove_bg_u2net(image):
    global u2net_model
    
    try:
        # Ensure model is loaded
        if u2net_model is None:
            u2net_model = initialize_u2net()
        
        if u2net_model is None:
            raise Exception("Failed to initialize U2Net model")
        
        # Convert OpenCV image to PIL
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Save original size
        original_size = img_pil.size
        
        # Resize for model input (U2Net works better with 512x512)
        input_size = 512  # Increased from 320 to 512 for better detail
        img_resized = img_pil.resize((input_size, input_size), Image.LANCZOS)
        
        # Normalize the image
        img_tensor = normalize(img_resized).to(device)
        
        # Predict mask
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = u2net_model(img_tensor)
            # Use d1 output (most detailed mask)
            pred = d1[:, 0, :, :]
            pred = torch.sigmoid(pred)
            pred = pred.data.cpu().numpy().squeeze()
        
        # Convert prediction to numpy array
        mask = (pred * 255).astype(np.uint8)
        
        # Resize mask back to original size
        mask_pil = Image.fromarray(mask).resize(original_size, Image.LANCZOS)
        mask = np.array(mask_pil)
        
        # Multi-level thresholding for clean foreground separation
        high_thresh = 240
        mid_thresh = 150
        low_thresh = 50
        
        mask_binary = np.zeros_like(mask)
        mask_binary[mask > high_thresh] = 255       # Definite foreground
        mask_binary[np.logical_and(mask >= mid_thresh, mask <= high_thresh)] = 200  # Likely foreground
        mask_binary[np.logical_and(mask >= low_thresh, mask < mid_thresh)] = 100    # Uncertain region
        mask_binary[mask < low_thresh] = 0          # Definite background
        
        # Apply connected components to clean up the mask
        num_labels, labels = cv2.connectedComponents(np.uint8(mask_binary > 0))
        
        # Find the largest component (likely to be the foreground)
        if num_labels > 1:
            areas = np.bincount(labels.flat)[1:]
            if len(areas) > 0:
                largest_label = np.argmax(areas) + 1
                mask_largest = np.uint8(labels == largest_label) * 255
                
                # Keep the largest component and uncertain regions
                mask_binary = np.maximum(mask_binary * (labels == largest_label), 
                                         np.uint8(mask >= low_thresh) * mask_binary)
        
        # Apply combined refinement methods for perfect edges
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # ADVANCED REFINEMENT: Multi-stage edge processing
        
        # 1. Create a trimap for alpha matting
        trimap = np.full(mask_binary.shape, 128, dtype=np.uint8)  # Unknown region
        trimap[mask_binary >= 200] = 255  # Definite foreground
        trimap[mask_binary == 0] = 0      # Definite background
        
        # 2. Apply guided filter for initial refinement
        refined_alpha = cv2.ximgproc.guidedFilter(
            guide=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            src=trimap.astype(np.float32) / 255.0,
            radius=8,
            eps=1e-7
        )
        
        # 3. Apply detail-preserving bilateral filtering
        refined_alpha = np.clip(refined_alpha * 255, 0, 255).astype(np.uint8)
        refined_alpha = cv2.bilateralFilter(refined_alpha, 7, 30, 30)
        
        # 4. Edge-aware refinement
        # Detect edges in the original image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # 5. Use joint bilateral filter to preserve edges from original image
        # Convert to float between 0-1 for processing
        mask_float = refined_alpha.astype(np.float32) / 255.0
        
        # Apply joint bilateral filter using original image as guide
        result_alpha = np.zeros_like(mask_float)
        for i in range(3):  # Multiple passes for better results
            result_alpha = cv2.ximgproc.jointBilateralFilter(
                joint=image,  # Original image guides the filtering
                src=mask_float if i == 0 else result_alpha,
                d=7,  # Diameter of each pixel neighborhood
                sigmaColor=50,  # Filter sigma in the color space
                sigmaSpace=50   # Filter sigma in the coordinate space
            )
        
        # 6. Final compositing with matting-based feathering
        # Create a narrow band around the edges for matting
        edge_band = cv2.dilate(np.uint8((result_alpha > 0.05) & (result_alpha < 0.95) * 255), 
                              np.ones((5,5), np.uint8), 
                              iterations=1)
        
        # Apply feathering only at the edge band
        feathered_alpha = result_alpha.copy()
        if np.any(edge_band > 0):
            # Apply extra smoothing at edges
            blurred_alpha = cv2.GaussianBlur(result_alpha, (5, 5), 0)
            feathered_alpha[edge_band > 0] = blurred_alpha[edge_band > 0]
        
        # Apply a subtle gradient to alpha values at object boundaries
        final_alpha = (feathered_alpha * 255).astype(np.uint8)
        
        # 7. Final soft decontamination step (removes color spill)
        rgba[:, :, 3] = final_alpha
        
        # For pixels with partial transparency, remove color contamination from background
        semi_transparent = (final_alpha > 0) & (final_alpha < 240)
        if np.any(semi_transparent):
            for c in range(3):  # For each color channel
                # Adjust color based on alpha value to reduce background contamination
                rgba[semi_transparent, c] = np.minimum(
                    255, 
                    rgba[semi_transparent, c] * (255.0 / np.maximum(1, rgba[semi_transparent, 3]))
                ).astype(np.uint8)
        
        return rgba
    except Exception as e:
        logger.error(f"Error in U2Net background removal: {e}")
        return None

# Advanced matting method for pixel-perfect edges
def refine_with_matting(image, alpha):
    try:
        # Convert to float
        alpha_f = alpha.astype(np.float32) / 255.0
        
        # Create trimap (foreground, background, and uncertain regions)
        trimap = np.ones_like(alpha) * 127  # Initialize all as uncertain
        trimap[alpha_f > 0.95] = 255  # Definite foreground
        trimap[alpha_f < 0.05] = 0    # Definite background
        
        # Apply alpha matting using KNN matting algorithm
        # (Simplified implementation of KNN matting)
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Resize for faster processing if image is large
        scale_factor = 1.0
        if max(h, w) > 1000:
            scale_factor = 1000.0 / max(h, w)
            h_scaled = int(h * scale_factor)
            w_scaled = int(w * scale_factor)
            image_small = cv2.resize(image, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
            trimap_small = cv2.resize(trimap, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)
        else:
            image_small = image
            trimap_small = trimap
        
        # Apply guided filter (as a simplified version of alpha matting)
        refined_alpha = cv2.ximgproc.guidedFilter(
            guide=cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY),
            src=trimap_small.astype(np.float32) / 255.0,
            radius=10,
            eps=1e-6
        )
        
        # Resize back to original size
        if scale_factor < 1.0:
            refined_alpha = cv2.resize(refined_alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to 8-bit
        refined_alpha = np.clip(refined_alpha * 255, 0, 255).astype(np.uint8)
        
        # Apply soft edge enhancement using bilateral filter
        refined_alpha = cv2.bilateralFilter(refined_alpha, 5, 50, 50)
        
        return refined_alpha
    except Exception as e:
        logger.error(f"Error in advanced matting: {e}")
        return alpha
# Enhanced edge refinement for pixel-perfect results
def enhanced_edge_refinement(image, alpha):
    try:
        # Convert to float
        alpha_f = alpha.astype(np.float32) / 255.0
        
        # Detect edges in alpha mask
        edges = cv2.Canny(alpha, 50, 150)
        edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        
        # Create trimap (foreground, background, and uncertain regions)
        trimap = np.ones_like(alpha) * 128  # Initialize all as uncertain
        trimap[alpha_f > 0.95] = 255  # Definite foreground
        trimap[alpha_f < 0.05] = 0    # Definite background
        
        # Make edge regions uncertain in trimap
        trimap[edge_dilated > 0] = 128
        
        # Apply multi-scale guided filtering
        refined_alpha = np.zeros_like(alpha_f)
        
        # Small radius for fine details
        refined_small = cv2.ximgproc.guidedFilter(
            guide=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            src=trimap.astype(np.float32) / 255.0,
            radius=4,
            eps=1e-6
        )
        
        # Medium radius for general edges
        refined_medium = cv2.ximgproc.guidedFilter(
            guide=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            src=trimap.astype(np.float32) / 255.0,
            radius=8,
            eps=1e-7
        )
        
        # Combine multi-scale results based on edge proximity
        edge_weight = cv2.distanceTransform(255 - edge_dilated, cv2.DIST_L2, 5)
        edge_weight = np.clip(edge_weight / 10.0, 0, 1)
        
        # Near edges: use fine-detail refinement, elsewhere use medium refinement
        refined_alpha = refined_small * (1 - edge_weight) + refined_medium * edge_weight
        
        # Convert back to 8-bit
        refined_alpha = np.clip(refined_alpha * 255, 0, 255).astype(np.uint8)
        
        # Apply detail preservation at hair-like edges
        # Detect potential hair/fur/fine details (high frequency edges)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        high_freq = cv2.subtract(gray, blur)
        high_freq_mask = (high_freq > 15).astype(np.uint8) * 255
        
        # Further refine these regions
        if np.any(high_freq_mask > 0):
            high_freq_mask = cv2.dilate(high_freq_mask, np.ones((3, 3), np.uint8), iterations=1)
            high_freq_regions = (high_freq_mask > 0) & (refined_alpha > 0) & (refined_alpha < 255)
            
            if np.any(high_freq_regions):
                # Apply local contrast enhancement to these regions
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                refined_alpha_clahe = clahe.apply(refined_alpha)
                refined_alpha[high_freq_regions] = refined_alpha_clahe[high_freq_regions]
        
        # Apply subtle anti-aliasing at edges
        edge_region = (refined_alpha > 5) & (refined_alpha < 250)
        if np.any(edge_region):
            edges_only = np.zeros_like(refined_alpha)
            edges_only[edge_region] = refined_alpha[edge_region]
            edges_aa = cv2.GaussianBlur(edges_only, (3, 3), 0.5)
            refined_alpha[edge_region] = edges_aa[edge_region]
        
        return refined_alpha
    except Exception as e:
        logger.error(f"Error in enhanced edge refinement: {e}")
        return alpha

# Combined background removal method
def remove_bg_perfect(image):
    try:
        # Pre-process image to improve segmentation
        # Normalize lighting and enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Try U2Net first (best quality)
        result = remove_bg_u2net(enhanced_img)
        
        # If U2Net fails, fall back to MediaPipe
        if result is None:
            logger.info("U2Net failed, falling back to MediaPipe")
            # Process with MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(rgb_image)
            
            # Get the segmentation mask
            mask = results.segmentation_mask
            mask = (mask * 255).astype(np.uint8)
            
            # Apply morphological operations to clean the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Create RGBA image
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = mask
        
        # Apply enhanced edge refinement for pixel-perfect results
        result[:, :, 3] = enhanced_edge_refinement(image, result[:, :, 3])
        
        # Final shadow and highlight correction
        # Detect semi-transparent regions (potential edge artifacts)
        semi_transparent = (result[:, :, 3] > 30) & (result[:, :, 3] < 230)
        
        if np.any(semi_transparent):
            # Detect darker regions that might be shadows and brighten them slightly
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dark_regions = (gray < 100) & semi_transparent
            
            if np.any(dark_regions):
                for c in range(3):
                    # Brighten dark edges slightly to reduce dark halos
                    result[dark_regions, c] = np.minimum(
                        255, 
                        (result[dark_regions, c] * 1.2).astype(np.uint8)
                    )
        
        return result
    except Exception as e:
        logger.error(f"Error in perfect background removal: {e}")
        # Final fallback - basic background removal
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask
        return result
# API to remove background from an image with perfect results
@app.route("/remove-bg", methods=["POST"])
def remove_bg_api():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        uploaded_file = request.files["image"]
        in_memory_file = io.BytesIO(uploaded_file.read())
        image_data = np.frombuffer(in_memory_file.getvalue(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Get the desired output format
        output_format = request.form.get("format", "png").lower()
        if output_format not in ["png", "webp"]:
            output_format = "png"  # Default to PNG
            
        # Get quality setting
        quality = request.form.get("quality", "high").lower()
        
        # Check if we should preserve original resolution
        preserve_resolution = request.form.get("preserve_resolution", "true").lower() == "true"
        
        # Image resizing strategy
        h, w = image.shape[:2]
        logger.info(f"Original image size: {w}x{h}")
        
        # For very large images, resize to a reasonable dimension
        MAX_DIMENSION = 1800 if quality == "high" else 1200  # Higher limit for high quality
        
        if max(h, w) > MAX_DIMENSION:
            # Calculate new dimensions maintaining aspect ratio
            if h > w:
                new_h = MAX_DIMENSION
                new_w = int(w * (MAX_DIMENSION / h))
            else:
                new_w = MAX_DIMENSION
                new_h = int(h * (MAX_DIMENSION / w))
                
            logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Process image with pixel-perfect background removal
        start_time = time.time()
        output_image = remove_bg_perfect(image)
        processing_time = time.time() - start_time
        logger.info(f"Background removal completed in {processing_time:.2f} seconds")
        
        if output_image is None:
            return jsonify({"error": "Failed to process image"}), 500

        # Output format handling with optimal settings
        if output_format == "webp":
            # WebP with optimal quality-size balance
            webp_quality = 95 if quality == "high" else 85
            _, buffer = cv2.imencode(".webp", output_image, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])
            mimetype = "image/webp"
            filename = "background_removed.webp"
        else:
            # PNG with appropriate compression
            compression_level = 1 if quality == "high" else 6  # Lower compression for high quality
            _, buffer = cv2.imencode(".png", output_image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
            mimetype = "image/png"
            filename = "background_removed.png"
        
        output_stream = io.BytesIO(buffer)
        output_stream.seek(0)
        
        logger.info(f"Final response size: {len(buffer)/1024/1024:.2f} MB")
        
        return send_file(
            output_stream,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.exception("Error in remove_bg_api")
        return jsonify({"error": str(e)}), 500
# @app.route("/remove-bg", methods=["POST"])
# def remove_bg_api():
#     try:
#         if "image" not in request.files:
#             return jsonify({"error": "No image file provided"}), 400

#         uploaded_file = request.files["image"]
#         in_memory_file = io.BytesIO(uploaded_file.read())
#         image_data = np.frombuffer(in_memory_file.getvalue(), np.uint8)
#         image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

#         if image is None:
#             return jsonify({"error": "Invalid image format"}), 400
        
#         # Get the desired output format
#         output_format = request.form.get("format", "png").lower()
#         if output_format not in ["png", "webp"]:
#             output_format = "png"  # Default to PNG
            
#         # Get quality setting
#         quality = request.form.get("quality", "high").lower()
        
#         # CRITICAL FIX: Force image resizing for very large images
#         h, w = image.shape[:2]
#         logger.info(f"Original image size: {w}x{h}")
        
#         # Set a hard limit on image dimensions
#         MAX_DIMENSION = 1200  # Maximum dimension allowed
        
#         if max(h, w) > MAX_DIMENSION:
#             # Calculate new dimensions maintaining aspect ratio
#             if h > w:
#                 new_h = MAX_DIMENSION
#                 new_w = int(w * (MAX_DIMENSION / h))
#             else:
#                 new_w = MAX_DIMENSION
#                 new_h = int(h * (MAX_DIMENSION / w))
                
#             logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
#             image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
#         # Process image with appropriate background removal
#         try:
#             output_image = remove_bg_perfect(image)
#         except Exception as e:
#             logger.error(f"Error in background removal: {e}")
#             # Fallback to basic background removal
#             output_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             output_image[:, :, 3] = mask
        
#         if output_image is None:
#             return jsonify({"error": "Failed to process image"}), 500

#         # CRITICAL FIX: Aggressive compression settings based on format
#         if output_format == "webp":
#             # WebP with moderate quality (adjust as needed)
#             webp_quality = 80 if quality == "high" else 60
#             _, buffer = cv2.imencode(".webp", output_image, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])
#             mimetype = "image/webp"
#             filename = "background_removed.webp"
#         else:
#             # PNG with higher compression
#             compression_level = 6 if quality == "high" else 9  # max compression
#             _, buffer = cv2.imencode(".png", output_image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
#             mimetype = "image/png"
#             filename = "background_removed.png"
        
#         output_stream = io.BytesIO(buffer)
#         output_stream.seek(0)
        
#         logger.info(f"Final response size: {len(buffer)/1024/1024:.2f} MB")
        
#         # CRITICAL FIX: Remove chunked transfer encoding
#         # Flask automatically handles the correct transfer encoding
#         response = send_file(
#             output_stream,
#             mimetype=mimetype,
#             as_attachment=True,
#             download_name=filename
#         )
        
#         # CRITICAL FIX: Add content-length but DO NOT add Transfer-Encoding header
#         response.headers["Content-Length"] = str(len(buffer))
        
#         return response

#     except Exception as e:
#         logger.exception("Error in remove_bg_api")
#         return jsonify({"error": str(e)}), 500

# Process video with transparent background using the improved method
def process_video_with_transparency(input_path, output_path, use_u2net=True):
    try:
        logger.info(f"Processing video: {input_path}")
        
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video file: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Calculate resize factor for processing (for efficiency)
        max_dim = 640  # Maximum dimension for video processing
        scale_factor = 1.0
        if max(width, height) > max_dim:
            scale_factor = max_dim / max(width, height)
            process_width = int(width * scale_factor)
            process_height = int(height * scale_factor)
        else:
            process_width = width
            process_height = height
        
        # Create an intermediate file for RGBA frames
        intermediate_path = output_path.replace(".mp4", "_alpha.avi")
        fourcc = cv2.VideoWriter_fourcc(*"RGBA")
        out_alpha = cv2.VideoWriter(intermediate_path, fourcc, fps, (width, height))
        
        if not out_alpha.isOpened():
            raise Exception(f"Error creating output video writer: {intermediate_path}")
        
        # Create the final output file
        fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
        out_final = cv2.VideoWriter(output_path, fourcc_mp4, fps, (width, height))
        
        if not out_final.isOpened():
            raise Exception(f"Error creating output video writer: {output_path}")
        
        frame_count = 0
        
        # Initialize MediaPipe for video processing
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for processing if needed
                if scale_factor < 1.0:
                    process_frame = cv2.resize(frame, (process_width, process_height), 
                                              interpolation=cv2.INTER_AREA)
                else:
                    process_frame = frame
                
                # Process the frame
                if use_u2net:
                    # Try U2Net for high quality
                    processed_frame = remove_bg_perfect(process_frame)
                else:
                    # Use MediaPipe for faster processing
                    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                    results = selfie_seg.process(rgb_frame)
                    mask = results.segmentation_mask
                    mask = (mask * 255).astype(np.uint8)
                    
                    # Create RGBA output
                    processed_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2BGRA)
                    processed_frame[:, :, 3] = mask
                
                # Resize back to original dimensions if needed
                if scale_factor < 1.0:
                    processed_frame = cv2.resize(processed_frame, (width, height), 
                                               interpolation=cv2.INTER_LANCZOS4)
                
                # Write to the transparent intermediate video
                out_alpha.write(processed_frame)
                
                # Also write to the final mp4 (without transparency)
                if processed_frame.shape[2] == 4:  # If RGBA
                    # Extract RGB channels
                    bgr_frame = processed_frame[:, :, 0:3]
                    # Get alpha channel and create mask
                    alpha = processed_frame[:, :, 3]
                    # Create a white background
                    white_bg = np.ones_like(bgr_frame) * 255
                    # Blend based on alpha
                    alpha_3channel = cv2.merge([alpha, alpha, alpha]) / 255.0
                    blended = (bgr_frame * alpha_3channel + white_bg * (1 - alpha_3channel)).astype(np.uint8)
                    out_final.write(blended)
                else:
                    out_final.write(processed_frame)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Release resources
        cap.release()
        out_alpha.release()
        out_final.release()
        
        logger.info(f"Video processing complete. Output saved to {output_path}")
        
        # Clean up intermediate file
        if os.path.exists(intermediate_path):
            os.remove(intermediate_path)
        
        # Verify final file exists
        if not os.path.exists(output_path):
            raise Exception(f"Output file was not created: {output_path}")
        
        return True
    except Exception as e:
        logger.exception(f"Error in process_video_with_transparency")
        return False

# API to remove background from a video with the improved method
@app.route("/remove-bg-video", methods=["POST"])
def remove_bg_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video = request.files["video"]
        filename = secure_filename(video.filename)

        if filename == "":
            return jsonify({"error": "Invalid file name"}), 400

        # Create absolute paths with timestamp
        timestamp = int(time.time())
        input_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{timestamp}.mp4")
        
        logger.info(f"Saving uploaded file to {input_path}")
        video.save(input_path)  # Save uploaded video file
        
        if not os.path.exists(input_path):
            return jsonify({"error": f"Failed to save upload file at {input_path}"}), 500
        
        # Get video quality parameter
        quality = request.form.get("quality", "medium").lower()
        
        # Use U2Net for high quality, MediaPipe for medium/low
        use_u2net = quality == "high"
        
        # Process video
        success = process_video_with_transparency(input_path, output_path, use_u2net=use_u2net)
        
        if not success:
            return jsonify({"error": "Failed to process video"}), 500
        
        if not os.path.exists(output_path):
            return jsonify({"error": f"Output file not found at {output_path}"}), 500
        
        # Get file size
        file_size = os.path.getsize(output_path)
        
        # Stream the file
        logger.info(f"Streaming file from {output_path} (size: {file_size} bytes)")
        
        # Function to stream file in chunks
        def generate_file_chunks(path, chunk_size=8192):
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        # Use streaming response
        return Response(
            stream_with_context(generate_file_chunks(output_path)),
            headers={
                "Content-Disposition": f"attachment; filename=bg_removed_{filename}",
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size)
            }
        )

    except Exception as e:
        logger.exception("Error in remove_bg_video")
        return jsonify({"error": str(e)}), 500

# Add a status endpoint
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "OK",
        "upload_folder": UPLOAD_FOLDER,
        "output_folder": OUTPUT_FOLDER,
        "model_folder": MODEL_FOLDER,
        "using_gpu": str(device) == "cuda"
    }), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)