from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from moviepy import VideoFileClip
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'single'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def modify_video(file_name, original_filename):
    """Apply phonk-style effects to video and return the processed file path"""
    import tempfile
    
    FILENAME = os.path.join("single", file_name)
    
    # Create a temporary file for the processed video
    temp_dir = tempfile.gettempdir()
    processed_filename = f"phonk_{original_filename}"
    to_write = os.path.join(temp_dir, processed_filename)

    clip = VideoFileClip(FILENAME)

    def apply_phonk_effects(frame):
        """Apply phonk-style visual effects to frame"""
        frame = frame.astype(np.float32)
        
        # Increase contrast (phonk style - high contrast)
        contrast = 1.4  # Increase contrast by 40%
        frame = ((frame - 128) * contrast + 128)
        
        # Adjust saturation
        # Convert to HSV-like adjustment
        saturation_boost = 1.3  # Increase saturation by 30%
        
        # Simple saturation boost by enhancing color differences from gray
        gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
        gray = np.expand_dims(gray, axis=2)
        
        # Boost color deviation from grayscale
        frame[...,:3] = gray + (frame[...,:3] - gray) * saturation_boost
        
        # Add slight purple/pink tint (typical phonk aesthetic)
        frame[..., 0] += 10  # Slight red boost
        frame[..., 2] += 15  # Purple/blue boost
        
        # Add vignette effect (darker edges)
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create vignette mask
        y, x = np.ogrid[:h, :w]
        mask = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = mask / np.max(mask)
        
        # Apply vignette (stronger effect)
        vignette_strength = 0.6
        vignette_mask = 1 - (mask * vignette_strength)
        vignette_mask = np.expand_dims(vignette_mask, axis=2)
        
        frame = frame * vignette_mask
        
        # Clamp values
        frame = np.clip(frame, 0, 255)
        
        return frame.astype(np.uint8)

    def optimized_transform(get_frame, t):
        fade_duration = 5.0
        
        reverse_t = clip.duration - t - 1/clip.fps
        reverse_t = max(0, min(reverse_t, clip.duration - 1/clip.fps))
        
        frame = get_frame(reverse_t)
        mirrored_frame = np.fliplr(frame)
        
        # Apply phonk effects
        phonk_frame = apply_phonk_effects(mirrored_frame)
        
        # Add subtle flicker effect (phonk aesthetic)
        flicker_intensity = 0.95 + 0.05 * np.sin(t * 30)  # Subtle flicker
        phonk_frame = (phonk_frame * flicker_intensity).astype(np.uint8)
        
        # Apply fade
        if t < fade_duration:
            alpha = t / fade_duration
            return (phonk_frame * alpha).astype(np.uint8)
        else:
            return phonk_frame

    final_clip = clip.transform(optimized_transform)

    # Apply audio effects for phonk style
    if clip.audio is not None:
        audio = clip.audio
        
        # Lower the pitch slightly and add some distortion-like effect
        # Note: For full phonk audio effects, you'd want to use additional audio libraries
        audio = audio.with_volume_scaled(0.8)  # Slightly lower volume
        final_clip = final_clip.with_audio(audio)

    final_clip = final_clip.with_duration(clip.duration)

    final_clip.write_videofile(
        to_write, 
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        preset="fast",  
        threads=4       
    )

    clip.close()
    final_clip.close()
    os.remove(FILENAME)  # Remove original uploaded file

    print(f"Phonk-style video processed: {processed_filename}")
    print(f"Removed original file: {file_name}")
    
    return to_write, processed_filename

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the post request has the file part
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename to avoid conflicts
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            
            # Save the uploaded file temporarily
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            print(f"File uploaded successfully: {unique_filename}")
            
            # Apply phonk effects to the video
            try:
                processed_file_path, processed_filename = modify_video(unique_filename, file.filename)
                
                # Send the processed file directly for download
                def cleanup_temp_file():
                    """Clean up temporary file after sending"""
                    try:
                        if os.path.exists(processed_file_path):
                            os.remove(processed_file_path)
                            print(f"Cleaned up temporary file: {processed_file_path}")
                    except Exception as e:
                        print(f"Cleanup error: {e}")
                
                # Return the file for download
                response = send_file(
                    processed_file_path,
                    as_attachment=True,
                    download_name=processed_filename,
                    mimetype='video/mp4'
                )
                
                # Schedule cleanup after response is sent
                import threading
                cleanup_thread = threading.Timer(2.0, cleanup_temp_file)
                cleanup_thread.start()
                
                return response
                
            except Exception as e:
                # Clean up uploaded file if processing fails
                if os.path.exists(filepath):
                    os.remove(filepath)
                print(f"Video processing error: {str(e)}")
                return jsonify({'error': f'Video processing failed: {str(e)}'}), 500
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WebM files.'}), 400
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Legacy download endpoint - now redirects to upload for direct download"""
    return jsonify({
        'message': 'Direct download is now handled through the upload endpoint',
        'info': 'Upload a video and it will be automatically downloaded after processing'
    }), 200

@app.route('/status')
def status():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Phonk video processor is ready',
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

@app.route('/')
def index():
    """Basic info endpoint"""
    return jsonify({
        'name': 'Phonk Video Processor API',
        'version': '1.0.0',
        'endpoints': {
            '/upload': 'POST - Upload and process video',
            '/download/<filename>': 'GET - Download processed video',
            '/status': 'GET - Check API status'
        }
    })

if __name__ == '__main__':
    print("Starting Phonk Video Processor Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print("Server running on http://localhost:5000")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)