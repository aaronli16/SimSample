from flask import (
    Flask, request, render_template, jsonify, send_from_directory,
    session, redirect, url_for
)
import os
import uuid
import shutil
from utils import audio_processing
import faiss
from werkzeug.utils import secure_filename
import zipfile
import tempfile
import subprocess
import platform
import json
from dotenv import load_dotenv
import gc
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required
)
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth

load_dotenv()

SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise RuntimeError('SECRET_KEY environment variable not set')

app = Flask(__name__, template_folder='./templates')
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB upload limit
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Simple health check endpoint for monitoring
@app.get('/health')
def health():
    return 'OK', 200

@app.route("/api/health-detailed")
def health_detailed():
    """Detailed health check to see what's working"""
    try:
        import sys
        import platform
        
        status = {
            "status": "ok",
            "python_version": sys.version,
            "platform": platform.system(),
            "dependencies": {}
        }
        
        # Test each dependency safely
        try:
            import librosa
            status["dependencies"]["librosa"] = "ok"
        except Exception as e:
            status["dependencies"]["librosa"] = f"error: {str(e)}"
        
        try:
            import faiss
            status["dependencies"]["faiss"] = "ok"  
        except Exception as e:
            status["dependencies"]["faiss"] = f"error: {str(e)}"
            
        try:
            import numpy
            status["dependencies"]["numpy"] = "ok"
        except Exception as e:
            status["dependencies"]["numpy"] = f"error: {str(e)}"
            
        # Test ffmpeg
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            status["dependencies"]["ffmpeg"] = "ok"
        except Exception as e:
            status["dependencies"]["ffmpeg"] = f"error: {str(e)}"
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
# --- Authentication setup ---
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
    raise RuntimeError('Google OAuth credentials not set')

oauth = OAuth(app)
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    client_kwargs={'scope': 'openid email profile'},
    # Manual endpoint configuration instead of discovery URL
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    access_token_url='https://oauth2.googleapis.com/token',
    userinfo_endpoint='https://www.googleapis.com/oauth2/v2/userinfo',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs'
)

USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

users = load_users()

class User(UserMixin):
    def __init__(self, id_, name, email):
        self.id = id_
        self.name = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    data = users.get(user_id)
    if data:
        return User(user_id, data.get('name'), data.get('email'))
    return None

# Base directories
UPLOAD_FOLDER = os.path.join("audio_samples", "uploads")
USER_LIBRARIES_FOLDER = os.path.join("audio_samples", "user_libraries")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_LIBRARIES_FOLDER, exist_ok=True)

# Store user sessions and their search systems
user_sessions = {}

def get_user_session():
    """Get or create a user session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session.permanent = True
    
    user_id = session['user_id']
    
    # Initialize user session if not exists
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'fingerprint_db': None,
            'faiss_index': None,
            'filenames': None,
            'library_folder': None,
            'library_built': False
        }
        
        # Try to load existing library from disk
        user_library_path = get_user_library_path(user_id)
        user_db_path = os.path.join(user_library_path, "fingerprint_db.pkl")
        user_index_path = os.path.join(user_library_path, "faiss_index.bin")
        user_names_path = os.path.join(user_library_path, "filenames.pkl")
        
        # Check if all library files exist
        if (os.path.exists(user_db_path) and 
            os.path.exists(user_index_path) and 
            os.path.exists(user_names_path) and
            os.path.exists(user_library_path)):
            
            try:
                print(f"🔄 Loading existing library for user {user_id[:8]}...")
                
                # Load the saved library data
                fingerprint_db, faiss_index, filenames = audio_processing.load_faiss_system(
                    user_db_path, user_index_path, user_names_path
                )
                
                # Restore to session
                user_sessions[user_id].update({
                    'fingerprint_db': fingerprint_db,
                    'faiss_index': faiss_index,
                    'filenames': filenames,
                    'library_folder': user_library_path,
                    'library_built': True
                })
                
                print(f"✅ Successfully loaded library with {len(fingerprint_db)} samples")
                
            except Exception as e:
                print(f"⚠️ Failed to load existing library: {e}")
                # If loading fails, keep the empty session (user can rebuild)
    
    return user_id

def get_user_library_path(user_id):
    """Get the path to user's library folder"""
    return os.path.join(USER_LIBRARIES_FOLDER, user_id)
@app.route("/api/warmup", methods=["GET", "POST"])
def warmup():
    """Simple session warmup"""
    try:
        user_id = get_user_session()
        return jsonify({"status": "ok", "user_id": user_id[:8]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --- Authentication routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        data = users.get(username)
        if data and check_password_hash(data['password'], password):
            user = User(username, data.get('name'), data.get('email'))
            login_user(user)
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('signup.html', error='User already exists')
        users[username] = {
            'password': generate_password_hash(password),
            'name': username,
            'email': username
        }
        save_users(users)
        user = User(username, username, username)
        login_user(user)
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login/google')
def login_google():
    redirect_uri = 'https://simsample-371783151021.us-central1.run.app/auth/google'
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/auth/google')
def google_authorize():
    try:
        token = oauth.google.authorize_access_token()
        if not token:
            print("No token received")
            return redirect(url_for('login'))
            
        # Use the full URL instead of just 'userinfo'
        resp = oauth.google.get('https://www.googleapis.com/oauth2/v2/userinfo', token=token)
        user_info = resp.json()
        
        print(f"User info received: {user_info}")
        
        user_id = user_info['email']
        users[user_id] = {
            'password': '',
            'name': user_info.get('name'),
            'email': user_info.get('email')
        }
        save_users(users)
        
        user = User(user_id, user_info.get('name'), user_info.get('email'))
        login_user(user)
        return redirect(url_for('index'))
        
    except Exception as e:
        print(f"OAuth error: {e}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('login'))
    
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/")
def index():
    return render_template("index.html")
# Replace these routes in your main.py

@app.route("/api/build-library", methods=["POST"])
def build_library():
    """Handle both single and chunked library uploads"""
    user_id = get_user_session()
    try:
        print("📋 BUILD: Starting build-library route")
        
        user_id = get_user_session()
        user_library_path = get_user_library_path(user_id)
        
        print(f"📋 BUILD: User ID: {user_id}")
        print(f"📋 BUILD: Library path: {user_library_path}")
        
        # Create user's library directory
        os.makedirs(user_library_path, exist_ok=True)
        
        # Handle different upload types
        if 'folder_zip' in request.files:
            print("📋 BUILD: Processing ZIP file")
            zip_file = request.files['folder_zip']
            if zip_file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Save and extract ZIP
            zip_path = os.path.join(user_library_path, secure_filename(zip_file.filename))
            zip_file.save(zip_path)
            
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(user_library_path)
                os.remove(zip_path)  # Clean up ZIP file
                print("📋 BUILD: ZIP extracted successfully")
                
            except zipfile.BadZipFile:
                return jsonify({"error": "Invalid ZIP file"}), 400
        
        elif 'folder_files' in request.files:
            print("📋 BUILD: Processing multiple files")
            files = request.files.getlist('folder_files')
            
            saved_count = 0
            audio_count = 0
            
            for file in files:
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    
                    # Check if it's an audio file
                    ext = filename.lower().split('.')[-1]
                    is_audio = ext in ['wav', 'mp3', 'flac', 'aiff', 'ogg', 'm4a']
                    
                    # Preserve folder structure from webkitRelativePath if available
                    relative_path = request.form.get(f'path_{files.index(file)}', filename)
                    file_path = os.path.join(user_library_path, relative_path)
                    
                    # Create directories if needed
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.save(file_path)
                    saved_count += 1
                    
                    if is_audio:
                        audio_count += 1
                    
            print(f"📋 BUILD: Saved {saved_count} files ({audio_count} audio files)")
        
        else:
            return jsonify({"error": "No files uploaded"}), 400
        
        # Count total audio files in library now
        try:
            all_audio_files = audio_processing.get_audio_files(user_library_path)
            total_audio = len(all_audio_files)
            print(f"📋 BUILD: Total audio files in library: {total_audio}")
            
            if total_audio == 0:
                return jsonify({"error": "No audio files found in uploaded content"}), 400
            
            return jsonify({
                "status": "uploaded",
                "message": f"Added files successfully. Library now contains {total_audio} audio files",
                "total_audio_files": total_audio,
                "chunk_files": saved_count if 'folder_files' in request.files else 1
            })
            
        except Exception as e:
            print(f"📋 BUILD: Error counting audio files: {e}")
            return jsonify({"error": f"Error processing files: {str(e)}"}), 500
        
    except Exception as e:
        import traceback
        print(f"❌ BUILD ERROR: {e}")
        print(f"📋 TRACEBACK: {traceback.format_exc()}")
        return jsonify({"error": f"Build failed: {str(e)}"}), 500


# Replace your current process_library route with these new chunked APIs:

# Replace your current process_library route with these new chunked APIs:

@app.route("/api/start-processing", methods=["POST"])
def start_processing():
    """Initialize batch processing - figure out how many batches we need"""
    try:
        user_id = get_user_session()
        user_library_path = get_user_library_path(user_id)

        if not os.path.exists(user_library_path):
            return jsonify({"error": "No library uploaded"}), 400

        # Get all audio files
        audio_files = audio_processing.get_audio_files(user_library_path)
        total_files = len(audio_files)

        if total_files == 0:
            return jsonify({"error": "No audio files found"}), 400

        # Calculate batches
        BATCH_SIZE = 20  # Process 20 files per batch
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

        # Initialize processing state in session
        user_sessions[user_id] = user_sessions.get(user_id, {})
        user_sessions[user_id].update({
            'processing_state': {
                'audio_files': audio_files,
                'total_files': total_files,
                'total_batches': total_batches,
                'current_batch': 0,
                'processed_files': 0,
                'fingerprint_db': {},
                'batch_size': BATCH_SIZE,
                'processing_started': True
            },
            'library_built': False  # Not ready yet
        })

        print(f"🚀 PROCESSING: Starting {total_files} files in {total_batches} batches")

        return jsonify({
            "status": "initialized",
            "total_files": total_files,
            "total_batches": total_batches,
            "batch_size": BATCH_SIZE
        })

    except Exception as e:
        print(f"❌ START ERROR: {e}")
        return jsonify({"error": f"Failed to start processing: {str(e)}"}), 500


@app.route("/api/process-library-batch", methods=["POST"])
def process_library_batch():
    """Process one batch of files"""
    import gc
    import time

    try:
        user_id = get_user_session()
        user_data = user_sessions.get(user_id, {})
        processing_state = user_data.get('processing_state')

        if not processing_state or not processing_state.get('processing_started'):
            return jsonify({"error": "Processing not initialized. Call /api/start-processing first"}), 400

        current_batch = processing_state['current_batch']
        total_batches = processing_state['total_batches']
        batch_size = processing_state['batch_size']
        audio_files = processing_state['audio_files']

        if current_batch >= total_batches:
            return jsonify({"error": "All batches already processed"}), 400

        # Get files for this batch
        start_idx = current_batch * batch_size
        end_idx = min(start_idx + batch_size, len(audio_files))
        batch_files = audio_files[start_idx:end_idx]
        batch_size_actual = len(batch_files)  # Store length for later use

        print(f"🔄 BATCH {current_batch + 1}/{total_batches}: Processing {batch_size_actual} files")

        # Process this batch
        batch_fingerprints = {}
        successful_files = 0

        for i, file_path in enumerate(batch_files):
            try:
                print(f"  📄 {i + 1}/{batch_size_actual}: {os.path.basename(file_path)}")
                fingerprint = audio_processing.generate_fingerprint(file_path)

                if fingerprint is not None and len(fingerprint) == 6:
                    batch_fingerprints[file_path] = fingerprint
                    successful_files += 1
                else:
                    print(f"    ⚠️ Invalid fingerprint")

                # Cleanup after each file
                del fingerprint
                gc.collect()

            except Exception as e:
                print(f"    ❌ Error processing {file_path}: {e}")
                continue

        # Add batch results to main database
        processing_state['fingerprint_db'].update(batch_fingerprints)
        processing_state['processed_files'] += successful_files
        processing_state['current_batch'] += 1

        # Aggressive cleanup
        del batch_fingerprints
        del batch_files
        gc.collect()

        print(f"✅ BATCH {current_batch + 1} complete: {successful_files} files processed")

        # Calculate progress
        progress = (processing_state['current_batch'] / total_batches) * 100

        return jsonify({
            "status": "batch_complete",
            "batch_number": current_batch + 1,
            "total_batches": total_batches,
            "files_in_batch": batch_size_actual,
            "successful_files": successful_files,
            "total_processed": processing_state['processed_files'],
            "total_files": processing_state['total_files'],
            "progress_percent": round(progress, 1),
            "is_final_batch": processing_state['current_batch'] >= total_batches
        })

    except Exception as e:
        print(f"❌ BATCH ERROR: {e}")
        import traceback
        print(f"📋 TRACEBACK: {traceback.format_exc()}")
        gc.collect()
        return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500


@app.route("/api/finalize-library", methods=["POST"])
def finalize_library():
    """Build final FAISS index and complete the library"""
    import gc
    import time

    try:
        user_id = get_user_session()
        user_data = user_sessions.get(user_id, {})
        processing_state = user_data.get('processing_state')

        if not processing_state:
            return jsonify({"error": "No processing state found"}), 400

        fingerprint_db = processing_state.get('fingerprint_db', {})
        if not fingerprint_db:
            return jsonify({"error": "No fingerprints to finalize"}), 400

        print(f"🔍 FINALIZE: Building FAISS index with {len(fingerprint_db)} samples")

        # Build FAISS index
        faiss_index, filenames = audio_processing.build_faiss_index(fingerprint_db)
        if faiss_index is None:
            return jsonify({"error": "Failed to build FAISS index"}), 500

        print(f"✅ FINALIZE: FAISS index built with {len(filenames)} samples")

        # Save everything
        user_library_path = user_data.get('library_folder') or get_user_library_path(user_id)
        db_path = os.path.join(user_library_path, "fingerprint_db.pkl")
        index_path = os.path.join(user_library_path, "faiss_index.bin")
        names_path = os.path.join(user_library_path, "filenames.pkl")

        audio_processing.save_faiss_system(
            fingerprint_db, faiss_index, filenames,
            db_path, index_path, names_path
        )

        # Update session with final state
        user_sessions[user_id].update({
            'fingerprint_db': fingerprint_db,
            'faiss_index': faiss_index,
            'filenames': filenames,
            'library_folder': user_library_path,
            'library_built': True,
            'processing_state': None  # Clear processing state
        })

        # Final cleanup
        gc.collect()

        print(f"🎉 FINALIZE: Library complete with {len(fingerprint_db)} samples!")

        return jsonify({
            "status": "completed",
            "sample_count": len(fingerprint_db),
            "message": f"Library ready with {len(fingerprint_db)} samples!"
        })

    except Exception as e:
        print(f"❌ FINALIZE ERROR: {e}")
        import traceback
        print(f"📋 TRACEBACK: {traceback.format_exc()}")
        gc.collect()
        return jsonify({"error": f"Finalization failed: {str(e)}"}), 500


@app.route("/api/processing-status")
def processing_status():
    """Get current processing progress"""
    user_id = get_user_session()
    user_data = user_sessions.get(user_id, {})
    processing_state = user_data.get('processing_state')

    if not processing_state:
        return jsonify({"processing": False, "library_built": user_data.get('library_built', False)})

    progress = (processing_state['current_batch'] / processing_state['total_batches']) * 100

    return jsonify({
        "processing": True,
        "current_batch": processing_state['current_batch'],
        "total_batches": processing_state['total_batches'],
        "processed_files": processing_state['processed_files'],
        "total_files": processing_state['total_files'],
        "progress_percent": round(progress, 1)
    })

def build_fingerprint_database_batched(folder_path, batch_size=5):  # ← DEFAULT TO 5
    """Build fingerprint database in small batches for large libraries"""
    import gc
    
    audio_files = audio_processing.get_audio_files(folder_path)
    
    # ADD THIS LINE - limit files for very large libraries:
    if len(audio_files) > 300:
        print(f"⚠️ Large library detected ({len(audio_files)} files). Processing first 300 files.")
        audio_files = audio_files[:300]
    
    fingerprint_database = {}
    
    total_files = len(audio_files)
    print(f"🔄 Processing {total_files} audio files in batches of {batch_size}...")
    
    # Process in batches
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        # Process each file in the batch
        for j, file_path in enumerate(batch):
            try:
                print(f"  📄 {j+1}/{len(batch)}: {os.path.basename(file_path)}")
                fingerprint = audio_processing.generate_fingerprint(file_path)
                
                if fingerprint is not None and len(fingerprint) == 6:
                    fingerprint_database[file_path] = fingerprint
                else:
                    print(f"    ⚠️ Skipping invalid fingerprint")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Force garbage collection after each batch
        gc.collect()
        
        # Show progress
        processed = min(i + batch_size, total_files)
        print(f"✅ Batch {batch_num} complete. Progress: {processed}/{total_files} files")
    
    # Final cleanup
    gc.collect()
    
    valid_count = len(fingerprint_database)
    print(f"🎉 Successfully processed {valid_count} out of {total_files} files")
    
    return fingerprint_database
@app.route("/api/library-status")
def library_status():
    """Get current library status for user"""
    user_id = get_user_session()
    user_data = user_sessions.get(user_id, {})
    
    # Handle None fingerprint_db properly
    fingerprint_db = user_data.get('fingerprint_db')
    sample_count = len(fingerprint_db) if fingerprint_db else 0
    
    return jsonify({
        "library_built": user_data.get('library_built', False),
        "sample_count": sample_count,
        "has_upload": user_data.get('library_folder') is not None
    })

@app.route("/api/library-stats")
def library_stats():
    """Return statistics about the current user's library"""
    user_id = get_user_session()
    user_data = user_sessions.get(user_id, {})

    if not user_data.get('library_built'):
        return jsonify({"error": "No library built"}), 400

    fingerprint_db = user_data.get('fingerprint_db')
    stats = audio_processing.get_library_stats(fingerprint_db)
    return jsonify(stats)

@app.route("/api/find-similar", methods=["POST"])
def find_similar():
    """Find similar samples in user's library"""
    user_id = get_user_session()
    user_data = user_sessions.get(user_id, {})

    # Check if user has built a library
    if not user_data.get('library_built'):
        return jsonify({"error": "Please upload and build your sample library first"}), 400

    # Ensure library is up to date on every search
    user_library_path = user_data['library_folder']
    db_path = os.path.join(user_library_path, "fingerprint_db.pkl")
    index_path = os.path.join(user_library_path, "faiss_index.bin")
    names_path = os.path.join(user_library_path, "filenames.pkl")

    fingerprint_db, faiss_index, filenames, refreshed = audio_processing.ensure_library_is_current(
        user_data['fingerprint_db'],
        user_data['faiss_index'],
        user_data['filenames'],
        user_library_path,
        db_path,
        index_path,
        names_path,
    )

    # Persist in session
    user_sessions[user_id]['fingerprint_db'] = fingerprint_db
    user_sessions[user_id]['faiss_index'] = faiss_index
    user_sessions[user_id]['filenames'] = filenames

    if "sample" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["sample"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploaded sample
    filepath = os.path.join(UPLOAD_FOLDER, f"{user_id}_{file.filename}")
    file.save(filepath)
    
    # Generate fingerprint for uploaded sample
    fingerprint = audio_processing.generate_fingerprint(filepath)
    if fingerprint is None:
        return jsonify({"error": "Could not process audio file"}), 400
    
    # Use user's FAISS index for search
    matches = audio_processing.find_similar_faiss(
        fingerprint, 
        user_data['faiss_index'], 
        user_data['filenames'],
        uploaded_filename=file.filename,
        filter_by_type=True
    )
    
    # Format results
    results = []
    user_library_path = user_data['library_folder']
    
    for path, score in matches:
        try:
            rel_path = os.path.relpath(path, user_library_path)
            filename = os.path.basename(path)
            
            sample_type = audio_processing.classify_sample_from_filename(filename)
            bpm = audio_processing.extract_bpm_from_filename(filename)
            key = audio_processing.extract_key_from_filename(filename)
            
            from urllib.parse import quote
            encoded_path = quote(f"{user_id}/{rel_path}".replace('\\', '/'))
            
            results.append({
                "path": path,
                "filename": filename,
                "score": float(score),
                "url": f"/api/user-samples/{encoded_path}",
                "type": sample_type,
                "bpm": bpm

            })
        except Exception as e:
            print(f"Error processing path {path}: {e}")
            continue
    
    return jsonify({
        "uploaded": file.filename,
        "uploadedUrl": f"/api/uploads/{user_id}_{file.filename}",
        "matches": results,
        "libraryRefreshed": refreshed
    })

# Serve user's sample files
@app.route("/api/user-samples/<path:filename>")
def serve_user_sample(filename):
    """Serve audio files from user's library"""
    from urllib.parse import unquote
    decoded_filename = unquote(filename)
    
    # Extract user_id from path
    path_parts = decoded_filename.split('/', 1)
    if len(path_parts) != 2:
        return "Invalid path", 400
    
    user_id, file_path = path_parts
    user_library_path = get_user_library_path(user_id)
    full_path = os.path.join(user_library_path, file_path)
    
    # Security check
    if not os.path.abspath(full_path).startswith(os.path.abspath(user_library_path)):
        return "Access denied", 403
    
    if os.path.exists(full_path):
        directory = os.path.dirname(full_path)
        filename_only = os.path.basename(full_path)
        return send_from_directory(directory, filename_only)
    else:
        return "File not found", 404

# Serve uploaded sample files
@app.route("/api/uploads/<path:filename>")
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/api/clear-library", methods=["POST"])
def clear_library():
    """Clear user's current library"""
    user_id = get_user_session()
    user_library_path = get_user_library_path(user_id)
    
    # Clear session data
    if user_id in user_sessions:
        user_sessions[user_id] = {
            'fingerprint_db': None,
            'faiss_index': None,
            'filenames': None,
            'library_folder': None,
            'library_built': False
        }
    
    # Remove library files
    if os.path.exists(user_library_path):
        shutil.rmtree(user_library_path)
    
    return jsonify({"status": "cleared"})


   
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=debug, host="0.0.0.0", port=port)


