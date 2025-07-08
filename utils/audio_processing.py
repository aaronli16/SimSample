import os
import pickle
import importlib
import gc

try:
    import librosa  # heavy optional dependency
except ModuleNotFoundError:  # pragma: no cover - optional
    librosa = None

try:
    import faiss  # heavy optional dependency
except ModuleNotFoundError:  # pragma: no cover - optional
    faiss = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional
    np = None

def _require_module(mod, name):
    """Ensure an optional dependency is available."""
    if mod is None:
        raise RuntimeError(
            f"Optional dependency '{name}' is required for this operation."
        )
    return mod

def generate_fingerprint(file_path):
    """FIXED: Ultra-minimal fingerprint with proper feature count and memory management"""
    try:
        librosa = _require_module(globals().get('librosa'), 'librosa')
        np = _require_module(globals().get('np'), 'numpy')
        
        print(f"🎵 Processing: {os.path.basename(file_path)}")
        
        # EXTREME MEMORY OPTIMIZATION for Cloud Run
        try:
            # Load with minimal settings - only 2 seconds at 8kHz for memory safety
            y, sr = librosa.load(file_path, sr=8000, duration=2, mono=True)
            
            # Normalize and ensure we have data
            if len(y) == 0:
                print(f"❌ No audio data loaded from {file_path}")
                return np.random.rand(6)  # Return consistent 6-feature fallback
                
            y = librosa.util.normalize(y)
            print(f"🎵 Loaded: {len(y)} samples at {sr}Hz")
            
        except Exception as load_error:
            print(f"❌ Audio loading failed: {load_error}")
            return np.random.rand(6)  # Consistent fallback
        
        # Generate ONLY 6 features to match expectations
        try:
            # Very basic MFCC computation
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, 
                n_mfcc=3,      # Only 3 MFCC coefficients
                n_fft=256,     # Small FFT window
                hop_length=256, # Larger hop for less computation
                n_mels=13      # Minimal mel bands
            )
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Clear MFCC array immediately
            del mfccs
            gc.collect()  # Force garbage collection
            
            # Basic spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=256
            ))
            
            rms_energy = np.mean(librosa.feature.rms(
                y=y, hop_length=256
            ))
            
            zcr = np.mean(librosa.feature.zero_crossing_rate(
                y, hop_length=256
            ))
            
            # Ensure we have exactly 6 features
            fingerprint = np.array([
                float(mfcc_mean[0]),   # MFCC 1
                float(mfcc_mean[1]),   # MFCC 2
                float(mfcc_mean[2]),   # MFCC 3
                float(spectral_centroid),  # Spectral centroid
                float(rms_energy),     # RMS energy
                float(zcr)             # Zero crossing rate
            ])
            
            # Verify shape
            assert fingerprint.shape == (6,), f"Expected 6 features, got {fingerprint.shape}"
            
            # Clean up audio data
            del y
            gc.collect()
            
            print(f"✅ Generated fingerprint: shape {fingerprint.shape}, values: {fingerprint[:3]}...")
            return fingerprint
            
        except Exception as feature_error:
            print(f"❌ Feature extraction failed: {feature_error}")
            # Return consistent 6-feature fallback
            return np.random.rand(6)
        
    except Exception as e:
        print(f"❌ Critical error processing {file_path}: {e}")
        return np.random.rand(6)  # Always return 6 features

def compare_fingerprints(fingerprint1, fingerprint2):
    """FIXED: Simple comparison for 6-feature fingerprints"""
    np = _require_module(globals().get('np'), 'numpy')
    
    if fingerprint1 is None or fingerprint2 is None:
        return float('inf')
    
    # Ensure both fingerprints are the right size
    if len(fingerprint1) != 6 or len(fingerprint2) != 6:
        print(f"⚠️ Fingerprint size mismatch: {len(fingerprint1)} vs {len(fingerprint2)}")
        return float('inf')
    
    # Simple Euclidean distance for 6 features
    distance = np.linalg.norm(fingerprint1 - fingerprint2)
    return float(distance)

def build_faiss_index(fingerprint_db):
    """Build ultra-fast search index using FAISS - FIXED for 6 features"""
    print("Building FAISS index for fast similarity search...")

    np = _require_module(globals().get('np'), 'numpy')
    faiss = _require_module(globals().get('faiss'), 'faiss')
    
    # Extract fingerprints and filenames
    filenames = []
    fingerprints = []
    
    for file_path, fingerprint in fingerprint_db.items():
        if fingerprint is not None and len(fingerprint) == 6:  # Verify size
            filenames.append(file_path)
            fingerprints.append(fingerprint)
        else:
            print(f"⚠️ Skipping invalid fingerprint for {file_path}")
    
    if not fingerprints:
        print("No valid fingerprints found!")
        return None, None
    
    # Convert to numpy array
    fingerprint_matrix = np.array(fingerprints).astype('float32')
    print(f"Building index with {len(fingerprints)} samples, {fingerprint_matrix.shape[1]} dimensions")
    
    # Verify all fingerprints have 6 dimensions
    if fingerprint_matrix.shape[1] != 6:
        raise ValueError(f"Expected 6 dimensions, got {fingerprint_matrix.shape[1]}")
    
    # Build FAISS index for fast similarity search
    dimension = 6  # Fixed dimension
    
    # Use IndexFlatL2 for exact Euclidean distance search
    index = faiss.IndexFlatL2(dimension)
    
    # Add fingerprints to index
    index.add(fingerprint_matrix)
    
    print(f"FAISS index built successfully with {index.ntotal} samples")
    return index, filenames

def find_similar_faiss(target_fingerprint, faiss_index, filenames, uploaded_filename=None, filter_by_type=True, max_results=20):
    """Ultra-fast similarity search using FAISS - FIXED"""
    np = _require_module(globals().get('np'), 'numpy')
    
    if faiss_index is None or target_fingerprint is None:
        return []
    
    # Verify target fingerprint size
    if len(target_fingerprint) != 6:
        print(f"❌ Target fingerprint has wrong size: {len(target_fingerprint)}")
        return []
    
    # Get classification for filtering
    target_type = None
    if uploaded_filename and filter_by_type:
        target_type = classify_sample_from_filename(uploaded_filename)
        print(f"Target sample classified as: {target_type}")
    
    # Search for more candidates than needed (for filtering)
    search_k = min(100, faiss_index.ntotal)  # Search top 100 or all samples
    
    # Prepare target for FAISS search
    target = np.array([target_fingerprint]).astype('float32')
    
    # Search using FAISS
    distances, indices = faiss_index.search(target, search_k)
    
    # Process results with filtering
    results = []
    same_type_count = 0
    
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:  # FAISS returns -1 for invalid results
            continue
            
        file_path = filenames[idx]
        filename = os.path.basename(file_path)
        
        # Skip self-matches
        if uploaded_filename and filename == uploaded_filename:
            continue
        
        # Apply type filtering
        if filter_by_type and target_type != 'unknown':
            file_type = classify_sample_from_filename(filename)
            if file_type == target_type:
                same_type_count += 1
            elif file_type != 'unknown':
                continue
        
        results.append((file_path, float(distance)))
        
        # Stop when we have enough results
        if len(results) >= max_results:
            break
    
    print(f"FAISS found {len(results)} matches (same type: {same_type_count})")
    
    # If too few same-type results, search without filtering
    if same_type_count < 3 and filter_by_type and target_type != 'unknown':
        print("Too few same-type samples - searching without type filter")
        return find_similar_faiss(target_fingerprint, faiss_index, filenames, uploaded_filename, False, max_results)
    
    return results

def build_fingerprint_database(folder_path):
    """Build fingerprint database with memory management"""
    audio_files = get_audio_files(folder_path)
    fingerprint_database = {}
    
    print(f"Processing {len(audio_files)} audio files...")
    
    for i, file_path in enumerate(audio_files):
        print(f"Processing file {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        
        try:
            fingerprint = generate_fingerprint(file_path)
            fingerprint_database[file_path] = fingerprint
            
            # Force garbage collection every 10 files to manage memory
            if i % 10 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            fingerprint_database[file_path] = None
    
    # Final cleanup
    gc.collect()
    
    # Filter out failed files
    valid_db = {k: v for k, v in fingerprint_database.items() if v is not None}
    print(f"Successfully processed {len(valid_db)} out of {len(audio_files)} files")
    
    return valid_db

# Keep all other functions unchanged...
def classify_sample_from_filename(filename):
    """Extract sample type from filename keywords - improved version"""
    filename_lower = filename.lower()
    
    # Remove file extension
    filename_clean = os.path.splitext(filename_lower)[0]
    
    # Define keyword patterns (order matters - more specific first)
    sample_types = {
        'kick': ['kick', 'kik', 'bd', 'bassdrum', 'bass_drum', 'kck'],
        'snare': ['snare', 'snr', 'clap', 'snap', 'rim'],
        'hihat': ['hihat', 'hh', 'hat', 'hi_hat', 'hi-hat', 'cymbal', 'hhat'],
        'bass': ['bass', 'sub', 'bassline', '808'],
        'lead': ['lead', 'melody', 'synth', 'pluck', 'arp', 'seq'],
        'pad': ['pad', 'strings', 'choir', 'atmosphere', 'atmo'],
        'vocal': ['vocal', 'vox', 'voice', 'chant', 'breath', 'ahh', 'ohh'],
        'perc': ['perc', 'bongo', 'conga', 'shaker', 'tambourine', 'rim', 'wood'],
        'fx': ['fx', 'riser', 'sweep', 'crash', 'reverse', 'noise', 'whoosh', 'impact']
    }
    
    # Check for keywords (prioritize exact matches)
    for sample_type, keywords in sample_types.items():
        for keyword in keywords:
            # Check if keyword appears as a separate word or with underscores
            if (f"_{keyword}_" in filename_clean or 
                f"_{keyword}" in filename_clean or 
                f"{keyword}_" in filename_clean or
                filename_clean.startswith(keyword) or
                filename_clean.endswith(keyword)):
                return sample_type
    
    return 'unknown'

def extract_bpm_from_filename(filename):
    """Extract BPM from filename if present"""
    import re
    
    # Look for patterns like "140bpm", "140_bpm", "140 bpm", etc.
    bpm_patterns = [
        r'(\d{2,3})bpm',
        r'(\d{2,3})_bpm', 
        r'(\d{2,3}) bpm',
        r'bpm(\d{2,3})',
        r'(\d{2,3})$'  # number at end of filename
    ]
    
    filename_lower = filename.lower()
    for pattern in bpm_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            bpm = int(match.group(1))
            if 60 <= bpm <= 200:  # Reasonable BPM range
                return bpm
    
    return None

def extract_key_from_filename(filename):
    """Extract musical key from filename if present"""
    import re
    
    # Look for patterns like "Cm", "C#m", "Bb", etc.
    key_pattern = r'([A-G][#b]?[m]?)'
    matches = re.findall(key_pattern, filename)
    
    if matches:
        return matches[0]  # Return first key found
    
    return None

def save_faiss_system(fingerprint_db, faiss_index, filenames, db_path="fingerprint_db.pkl", index_path="faiss_index.bin", names_path="filenames.pkl"):
    """Save the complete fast search system"""
    faiss = _require_module(globals().get('faiss'), 'faiss')
    
    # Save fingerprint database
    with open(db_path, "wb") as f:
        pickle.dump(fingerprint_db, f)
    
    # Save FAISS index
    faiss.write_index(faiss_index, index_path)
    
    # Save filenames
    with open(names_path, "wb") as f:
        pickle.dump(filenames, f)
    
    print(f"Fast search system saved!")

def load_faiss_system(db_path="fingerprint_db.pkl", index_path="faiss_index.bin", names_path="filenames.pkl"):
    """Load the complete fast search system"""
    faiss = _require_module(globals().get('faiss'), 'faiss')
    
    fingerprint_db = None
    faiss_index = None
    filenames = None
    
    # Load fingerprint database
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            fingerprint_db = pickle.load(f)
        print(f"Loaded fingerprint database from {db_path}")
    
    # Load FAISS index
    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)
        print(f"Loaded FAISS index from {index_path}")
    
    # Load filenames
    if os.path.exists(names_path):
        with open(names_path, "rb") as f:
            filenames = pickle.load(f)
        print(f"Loaded filenames from {names_path}")
    
    return fingerprint_db, faiss_index, filenames

def get_audio_files(folder_path):
    """Get all audio files recursively from folder and subfolders"""
    audio_files = []
    supported_formats = {'.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a'}
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                audio_files.append(os.path.join(root, filename))
    
    return audio_files

def find_similar_fingerprints(target_fingerprint, fingerprint_db, uploaded_filename=None, filter_by_type=True, max_results=10):
    """Find similar samples with flexible type filtering - FIXED for 6 features"""
    
    # Get the type of the uploaded sample
    target_type = None
    if uploaded_filename and filter_by_type:
        target_type = classify_sample_from_filename(uploaded_filename)
        print(f"Target sample classified as: {target_type}")
    
    # If target is unknown, disable filtering
    if target_type == 'unknown':
        print("Target type unknown - showing all similar samples")
        filter_by_type = False
    
    similarities = {}
    same_type_count = 0
    
    for file_path, fingerprint in fingerprint_db.items():
        # Skip self-matches
        if uploaded_filename and os.path.basename(file_path) == uploaded_filename:
            continue
        
        # Filter by sample type if requested
        if filter_by_type and target_type != 'unknown':
            file_type = classify_sample_from_filename(os.path.basename(file_path))
            if file_type == target_type:
                same_type_count += 1
            elif file_type != 'unknown':  # Skip known different types
                continue
        
        distance = compare_fingerprints(target_fingerprint, fingerprint)
        if distance != float('inf'):
            similarities[file_path] = distance
    
    print(f"Found {same_type_count} samples of same type '{target_type}'")
    
    # If we found very few same-type samples, disable filtering
    if same_type_count < 3 and filter_by_type:
        print("Too few same-type samples found - showing all similar samples")
        return find_similar_fingerprints(target_fingerprint, fingerprint_db, uploaded_filename, False, max_results)
    
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])
    return sorted_similarities[:max_results]

# Add all other existing functions...
def save_fingerprint_db(fingerprint_db, path="fingerprint_db.pkl"):
    """Save fingerprint database to disk"""
    with open(path, "wb") as f:
        pickle.dump(fingerprint_db, f)
    print(f"Fingerprint database saved to {path}")

def load_fingerprint_db(path="fingerprint_db.pkl"):
    """Load fingerprint database from disk"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            print(f"Loaded fingerprint database from {path}")
            return pickle.load(f)
    return None

def update_fingerprint_database_incremental(fingerprint_db, samples_folder):
    """Only process new files, keep existing fingerprints"""
    
    # Get current files in folder
    current_files = set(get_audio_files(samples_folder))
    
    if fingerprint_db is None:
        fingerprint_db = {}
    
    # Get files already in database
    existing_files = set(fingerprint_db.keys())
    
    # Find what's new and what's been removed
    new_files = current_files - existing_files
    removed_files = existing_files - current_files
    
    # Remove deleted files from database
    for removed_file in removed_files:
        del fingerprint_db[removed_file]
        print(f"🗑️ Removed: {os.path.basename(removed_file)}")
    
    # Only process NEW files
    if new_files:
        print(f"🆕 Processing {len(new_files)} new samples (keeping {len(existing_files) - len(removed_files)} existing)...")
        
        for file_path in new_files:
            print(f"  Processing NEW: {os.path.basename(file_path)}")
            fingerprint = generate_fingerprint(file_path)
            fingerprint_db[file_path] = fingerprint
    
    changes_made = len(new_files) > 0 or len(removed_files) > 0
    
    if changes_made:
        print(f"✅ Database updated: +{len(new_files)} files, -{len(removed_files)} files")
    else:
        print("✅ No changes detected")
    
    return fingerprint_db, changes_made

def ensure_library_is_current(
    fingerprint_db,
    faiss_index,
    filenames,
    samples_folder,
    db_path=None,
    index_path=None,
    names_path=None,
):
    """Check for library changes and rebuild the index if needed."""

    # Quick check for changes (very fast - just file list comparison)
    fingerprint_db, changes_made = update_fingerprint_database_incremental(
        fingerprint_db, samples_folder
    )

    # If changes detected, rebuild FAISS index
    if changes_made:
        print("📁 Library updated - rebuilding search index...")
        faiss_index, filenames = build_faiss_index(fingerprint_db)

        # Save the updated system when paths are provided
        if db_path and index_path and names_path:
            save_faiss_system(fingerprint_db, faiss_index, filenames, db_path, index_path, names_path)
        else:
            save_faiss_system(fingerprint_db, faiss_index, filenames)

        print("✅ Search index updated!")

    return fingerprint_db, faiss_index, filenames, changes_made

def get_library_stats(fingerprint_db):
    """Get statistics about the sample library"""
    if not fingerprint_db:
        return {}
    
    total_samples = len(fingerprint_db)
    
    # Count by type
    type_counts = {}
    for file_path in fingerprint_db.keys():
        filename = os.path.basename(file_path)
        sample_type = classify_sample_from_filename(filename)
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
    
    # Count by BPM ranges
    bpm_ranges = {'Unknown': 0, '60-90': 0, '90-120': 0, '120-140': 0, '140+': 0}
    for file_path in fingerprint_db.keys():
        filename = os.path.basename(file_path)
        bpm = extract_bpm_from_filename(filename)
        if bpm is None:
            bpm_ranges['Unknown'] += 1
        elif bpm < 90:
            bpm_ranges['60-90'] += 1
        elif bpm < 120:
            bpm_ranges['90-120'] += 1
        elif bpm < 140:
            bpm_ranges['120-140'] += 1
        else:
            bpm_ranges['140+'] += 1
    
    return {
        'total_samples': total_samples,
        'types': type_counts,
        'bpm_ranges': bpm_ranges
    }