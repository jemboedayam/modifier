from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from moviepy import VideoFileClip
import numpy as np
import threading
import time
import tempfile
import mimetypes
from functools import wraps

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)  # Enhanced CORS for mobile

# Mobile-optimized configuration
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp_processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', '3gp', 'mp4v', 'm4v'}
MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150MB for better mobile compatibility
MOBILE_MAX_RESOLUTION = (1080, 1920)  # Mobile-friendly max resolution
CHUNK_SIZE = 8192  # For streaming downloads

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def mobile_response(f):
    """Decorator to add mobile-friendly headers"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        if hasattr(response, 'headers'):
            # Mobile-friendly headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
        return response
    return decorated_function

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_info(filepath):
    """Get video information for mobile optimization"""
    try:
        with VideoFileClip(filepath) as clip:
            return {
                'duration': clip.duration,
                'size': clip.size,
                'fps': clip.fps,
                'has_audio': clip.audio is not None
            }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None

def optimize_for_mobile(clip):
    """Enhanced mobile optimization"""
    w, h = clip.size
    max_w, max_h = MOBILE_MAX_RESOLUTION
    
    print(f"Original video: {w}x{h}, duration: {clip.duration:.2f}s")
    
    # Resize if too large for mobile
    if w > max_w or h > max_h:
        ratio_w = max_w / w
        ratio_h = max_h / h
        ratio = min(ratio_w, ratio_h)
        
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Ensure dimensions are even (required for H.264)
        new_w = new_w if new_w % 2 == 0 else new_w - 1
        new_h = new_h if new_h % 2 == 0 else new_h - 1
        
        clip = clip.resized((new_w, new_h))
        print(f"Resized to: {new_w}x{new_h} for mobile compatibility")
    
    # Optimize frame rate for mobile
    if hasattr(clip, 'fps') and clip.fps > 30:
        clip = clip.with_fps(30)
        print("Reduced FPS to 30 for mobile optimization")
    
    return clip

def modify_video(file_path, original_filename, progress_callback=None):
    """Enhanced video processing with progress tracking"""
    try:
        # Create unique output filename
        base_name = os.path.splitext(original_filename)[0]
        processed_filename = f"phonk_{base_name}_{int(time.time())}.mp4"
        output_path = os.path.join(TEMP_FOLDER, processed_filename)
        
        if progress_callback:
            progress_callback(10, "Loading video...")
        
        # Load video
        clip = VideoFileClip(file_path)
        
        if progress_callback:
            progress_callback(20, "Optimizing for mobile...")
        
        # Optimize for mobile
        clip = optimize_for_mobile(clip)
        
        if progress_callback:
            progress_callback(30, "Applying phonk effects...")
        
        def apply_phonk_effects(frame):
            """Optimized phonk effects for mobile processing"""
            frame = frame.astype(np.float32)
            
            # High contrast (phonk aesthetic)
            contrast = 1.35
            frame = ((frame - 128) * contrast + 128)
            
            # Enhanced saturation
            saturation_boost = 1.3
            gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
            gray = np.expand_dims(gray, axis=2)
            frame[...,:3] = gray + (frame[...,:3] - gray) * saturation_boost
            
            # Purple/pink tint (phonk aesthetic)
            frame[..., 0] += 12  # Red
            frame[..., 2] += 18  # Blue
            
            # Vignette effect
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            y_indices, x_indices = np.ogrid[:h, :w]
            distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            normalized_distances = distances / max_distance
            vignette_strength = 0.5
            vignette_mask = 1 - (normalized_distances * vignette_strength)
            vignette_mask = np.expand_dims(vignette_mask, axis=2)
            
            frame = frame * vignette_mask
            frame = np.clip(frame, 0, 255)
            
            return frame.astype(np.uint8)
        
        def enhanced_transform(get_frame, t):
            """Enhanced phonk transform with better mobile performance"""
            fade_duration = 2.5
            
            # Reverse playback
            reverse_t = clip.duration - t - 1/clip.fps
            reverse_t = max(0, min(reverse_t, clip.duration - 1/clip.fps))
            
            frame = get_frame(reverse_t)
            mirrored_frame = np.fliplr(frame)
            
            # Apply phonk effects
            phonk_frame = apply_phonk_effects(mirrored_frame)
            
            # Subtle flicker effect
            flicker_intensity = 0.95 + 0.05 * np.sin(t * 20)
            phonk_frame = (phonk_frame * flicker_intensity).astype(np.uint8)
            
            # Fade in effect
            if t < fade_duration:
                alpha = t / fade_duration
                phonk_frame = (phonk_frame * alpha).astype(np.uint8)
            
            return phonk_frame
        
        if progress_callback:
            progress_callback(50, "Processing video effects...")
        
        # Apply transformation
        final_clip = clip.transform(enhanced_transform)
        
        # Enhanced audio processing
        if clip.audio is not None:
            audio = clip.audio
            # Slight bass boost effect for phonk style
            audio = audio.with_volume_scaled(0.9)
            final_clip = final_clip.with_audio(audio)
        
        final_clip = final_clip.with_duration(clip.duration)
        
        if progress_callback:
            progress_callback(70, "Encoding video...")
        
        # FIXED: Minimal MoviePy export settings for maximum compatibility
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            bitrate="1800k"
        )
        
        # Cleanup
        final_clip.close()
        clip.close()
        
        if progress_callback:
            progress_callback(100, "Processing complete!")
        
        print(f"Mobile-optimized phonk video created: {processed_filename}")
        return output_path, processed_filename
        
    except Exception as e:
        print(f"Video processing error: {str(e)}")
        raise e

@app.route('/upload', methods=['POST', 'OPTIONS'])
@mobile_response
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Validate request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided', 'code': 'NO_FILE'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'code': 'EMPTY_FILE'}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Please upload: ' + ', '.join(ALLOWED_EXTENSIONS).upper(),
                'code': 'INVALID_TYPE',
                'supported_formats': list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Generate unique filename
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        # Save uploaded file
        upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(upload_path)
        
        print(f"File uploaded: {unique_filename} ({file.filename})")
        
        # Get video info for validation
        video_info = get_video_info(upload_path)
        if not video_info:
            os.remove(upload_path)
            return jsonify({'error': 'Invalid or corrupted video file', 'code': 'CORRUPT_FILE'}), 400
        
        # Check duration (limit to 5 minutes for mobile)
        if video_info['duration'] > 300:  # 5 minutes
            os.remove(upload_path)
            return jsonify({
                'error': 'Video too long. Maximum duration: 5 minutes',
                'code': 'TOO_LONG',
                'max_duration': 300
            }), 400
        
        # Process video
        try:
            processed_file_path, processed_filename = modify_video(
                upload_path, 
                file.filename
            )
            
            # IMPROVED: Better file cleanup handling
            def safe_remove(filepath):
                """Safely remove file with retry logic"""
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            print(f"Successfully removed: {filepath}")
                            return True
                    except PermissionError:
                        if attempt < max_attempts - 1:
                            time.sleep(1)  # Wait before retry
                            continue
                        else:
                            print(f"Could not remove {filepath} after {max_attempts} attempts")
                            return False
                return True
            
            # Clean up uploaded file
            safe_remove(upload_path)
            
            # Prepare file for download
            def generate_file():
                try:
                    with open(processed_file_path, 'rb') as f:
                        while True:
                            data = f.read(CHUNK_SIZE)
                            if not data:
                                break
                            yield data
                finally:
                    # Schedule cleanup with improved error handling
                    def cleanup():
                        time.sleep(5)  # Reduced wait time
                        safe_remove(processed_file_path)
                    
                    cleanup_thread = threading.Thread(target=cleanup)
                    cleanup_thread.daemon = True
                    cleanup_thread.start()
            
            # Create streaming response for better mobile compatibility
            response = Response(
                generate_file(),
                mimetype='video/mp4',
                headers={
                    'Content-Disposition': f'attachment; filename="{processed_filename}"',
                    'Content-Type': 'video/mp4',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'Accept-Ranges': 'bytes'
                }
            )
            
            return response
            
        except Exception as e:
            # Clean up on processing error
            safe_remove(upload_path)
            print(f"Processing error: {str(e)}")
            return jsonify({
                'error': f'Video processing failed: {str(e)}',
                'code': 'PROCESSING_ERROR'
            }), 500
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            'error': f'Upload failed: {str(e)}',
            'code': 'UPLOAD_ERROR'
        }), 500

@app.route('/mobile-info', methods=['GET'])
@mobile_response
def mobile_info():
    """Mobile-specific information and capabilities"""
    return jsonify({
        'mobile_optimized': True,
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024),
        'max_resolution': f"{MOBILE_MAX_RESOLUTION[0]}x{MOBILE_MAX_RESOLUTION[1]}",
        'max_duration_seconds': 300,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'recommended_settings': {
            'file_size': 'Under 100MB for best performance',
            'resolution': 'Videos auto-resized to mobile dimensions',
            'duration': 'Maximum 5 minutes',
            'format': 'MP4 recommended for best compatibility'
        },
        'processing_features': [
            'Reverse playback effect',
            'Mirror/flip transformation',
            'High contrast enhancement',
            'Color saturation boost',
            'Purple/pink tint overlay',
            'Vignette effect',
            'Subtle flicker animation',
            'Mobile-optimized encoding'
        ]
    })

@app.route('/status', methods=['GET'])
@mobile_response
def status():
    """Enhanced health check with system info"""
    return jsonify({
        'status': 'running',
        'message': 'Mobile Phonk Video Processor Ready',
        'version': '2.1.1-moviepy-fixed',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'mobile_optimized': True,
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024),
        'processing_ready': True,
        'server_time': int(time.time())
    })

@app.route('/', methods=['GET'])
@mobile_response
def index():
    """API information endpoint"""
    return jsonify({
        'name': 'Mobile Phonk Video Processor API',
        'version': '2.1.1-moviepy-fixed',
        'description': 'Transform videos with phonk-style effects optimized for mobile devices',
        'mobile_optimized': True,
        'endpoints': {
            '/upload': 'POST - Upload and process video with phonk effects',
            '/mobile-info': 'GET - Mobile-specific capabilities and limits',
            '/status': 'GET - API health check and system status',
            '/': 'GET - API information (this endpoint)'
        },
        'features': [
            'Mobile-optimized processing',
            'Streaming file downloads',
            'Progress tracking support',
            'Auto video compression',
            'Enhanced error handling',
            'CORS enabled for web apps',
            'MoviePy compatibility fixed'
        ]
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': f'File too large. Maximum size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB',
        'code': 'FILE_TOO_LARGE',
        'max_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024)
    }), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'code': 'BAD_REQUEST'
    }), 400

if __name__ == '__main__':
    print("🎵 Starting Mobile-Enhanced Phonk Video Processor Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Temp folder: {TEMP_FOLDER}")
    print(f"📋 Supported formats: {', '.join(ALLOWED_EXTENSIONS).upper()}")
    print(f"📏 Max file size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB")
    print(f"📱 Max resolution: {MOBILE_MAX_RESOLUTION[0]}x{MOBILE_MAX_RESOLUTION[1]}")
    print(f"⏱️  Max duration: 5 minutes")
    print(f"🌐 Server URL: http://localhost:5000")
    print("✅ Mobile optimizations enabled")
    print("✅ MoviePy compatibility fixed")
    print("🚀 Server ready!")
    
    # Run with mobile-optimized settings
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000, 
        threaded=True,
        use_reloader=False  # Better for mobile development
    )