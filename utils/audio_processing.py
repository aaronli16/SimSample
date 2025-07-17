import os
import pickle
import gc
import numpy as np
FINGERPRINT_DIMENSIONS = 12
try:
    import librosa
except ModuleNotFoundError:
    librosa = None

try:
    import faiss
except ModuleNotFoundError:
    faiss = None


def _require_module(mod, name):
    """Ensure an optional dependency is available."""
    if mod is None:
        raise RuntimeError(f"Optional dependency '{name}' is required for this operation.")
    return mod


def generate_fingerprint(file_path):
    """My honest expert recommendation - optimized for one-shot samples"""
    try:
        librosa_lib = _require_module(librosa, 'librosa')
        np_lib = _require_module(np, 'numpy')

        print(f"🎵 Processing: {os.path.basename(file_path)}")

        # OPTIMAL PARAMETERS for speed vs quality
        y, sr = librosa_lib.load(
            file_path,
            sr=11025,  # Sweet spot for analysis speed vs quality
            duration=2.5,  # Enough for one-shots, not wasteful
            mono=True
        )

        if len(y) == 0:
            return np_lib.zeros(FINGERPRINT_DIMENSIONS, dtype='float32')

        y = librosa_lib.util.normalize(y)
        features = []

        # === CORE FEATURES (8 total) ===

        # 1. MFCCs - The bread and butter (4 features)
        mfccs = librosa_lib.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=2,  # First 2 are most important
            n_fft=512,  # Good balance
            hop_length=256,
            n_mels=13
        )
        features.extend(np_lib.mean(mfccs, axis=1))  # 2 features
        features.extend(np_lib.std(mfccs, axis=1))  # 2 features

        # 2. Spectral centroid - Brightness (2 features)
        spectral_centroid = librosa_lib.feature.spectral_centroid(
            y=y, sr=sr, hop_length=256
        )
        features.append(np_lib.mean(spectral_centroid))
        features.append(np_lib.std(spectral_centroid))

        # 3. Zero crossing rate - Transient detection (1 feature)
        zcr = librosa_lib.feature.zero_crossing_rate(y, hop_length=256)
        features.append(np_lib.mean(zcr))

        # 4. RMS energy - Loudness characteristic (1 feature)
        rms = librosa_lib.feature.rms(y=y, hop_length=256)
        features.append(np_lib.mean(rms))

        # === ADVANCED FEATURES (4 total) ===

        # 5. Spectral rolloff - Frequency distribution (1 feature)
        spectral_rolloff = librosa_lib.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=256, roll_percent=0.85
        )
        features.append(np_lib.mean(spectral_rolloff))

        # 6. Spectral flux - Transient detection (BETTER than H/P for speed) (1 feature)
        stft = librosa_lib.stft(y, n_fft=512, hop_length=256)
        spectral_flux = np_lib.mean(np_lib.diff(np_lib.abs(stft), axis=1) ** 2)
        features.append(min(spectral_flux, 1.0))  # Normalize

        # 7. Spectral bandwidth - Frequency spread (1 feature)
        spectral_bandwidth = librosa_lib.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=256
        )
        features.append(np_lib.mean(spectral_bandwidth))

        # 8. Chroma energy - Harmonic content estimation (1 feature)
        chroma = librosa_lib.feature.chroma_stft(y=y, sr=sr, hop_length=256)
        chroma_energy = np_lib.sum(np_lib.var(chroma, axis=1))
        features.append(min(chroma_energy, 1.0))  # Normalize

        # === FINALIZATION ===
        features = np_lib.array(features[:FINGERPRINT_DIMENSIONS], dtype=np_lib.float32)
        features = np_lib.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        print(f"✅ Generated 12-feature fingerprint")
        return features

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return np.zeros(FINGERPRINT_DIMENSIONS, dtype='float32')


def compare_fingerprints(fingerprint1, fingerprint2):
    """Enhanced comparison for 12-feature fingerprints"""
    np_lib = _require_module(np, 'numpy')

    if fingerprint1 is None or fingerprint2 is None:
        return float('inf')

    if len(fingerprint1) != FINGERPRINT_DIMENSIONS or len(fingerprint2) != FINGERPRINT_DIMENSIONS:
        print(f"⚠️ Fingerprint size mismatch: {len(fingerprint1)} vs {len(fingerprint2)}")
        return float('inf')

    # Euclidean distance for 12 features
    distance = np_lib.linalg.norm(fingerprint1 - fingerprint2)
    return float(distance)


def build_faiss_index(fingerprint_db):
    """Build FAISS index for 12-dimensional features"""
    print("Building enhanced FAISS index...")

    np_lib = _require_module(np, 'numpy')
    faiss_lib = _require_module(faiss, 'faiss')

    filenames = []
    fingerprints = []

    for file_path, fingerprint in fingerprint_db.items():
        if fingerprint is not None and len(fingerprint) == FINGERPRINT_DIMENSIONS:
            filenames.append(file_path)
            fingerprints.append(fingerprint)
        else:
            print(f"⚠️ Skipping invalid fingerprint for {file_path}")

    if not fingerprints:
        print("No valid fingerprints found!")
        return None, None

    # Convert to numpy array
    fingerprint_matrix = np_lib.array(fingerprints).astype('float32')
    print(f"Building index with {len(fingerprints)} samples, {fingerprint_matrix.shape[1]} dimensions")

    # Build FAISS index
    dimension = FINGERPRINT_DIMENSIONS
    index = faiss_lib.IndexFlatL2(dimension)
    index.add(fingerprint_matrix)

    print(f"Enhanced FAISS index built with {index.ntotal} samples")
    return index, filenames


def calculate_adaptive_k_linear(library_size):
    """
    Linear scaling: +3 results for every 100 samples
    Perfect for one-shot sample libraries
    """
    # Handle very small libraries
    if library_size < 10:
        return max(1, library_size - 1)

    # Base results for small libraries
    base_results = 4
    baseline_samples = 50

    if library_size <= baseline_samples:
        return min(base_results, library_size - 1)

    # Linear scaling: +3 results per 100 samples above baseline
    additional_samples = library_size - baseline_samples
    additional_results = (additional_samples // 100) * 3

    # Add partial scaling for incomplete hundreds
    partial_hundred = additional_samples % 100
    if partial_hundred >= 50:  # If more than halfway, add 1-2 more
        additional_results += min(2, partial_hundred // 33)

    total_results = base_results + additional_results

    # Reasonable caps
    max_results = min(25, library_size // 5)  # Never more than 20% of library
    final_results = min(total_results, max_results)

    return final_results


def find_similar_faiss_linear_adaptive(target_fingerprint, faiss_index, filenames, uploaded_filename=None,
                                       filter_by_type=True):
    """Optimized similarity search with linear adaptive results"""
    np_lib = _require_module(np, 'numpy')

    if faiss_index is None or target_fingerprint is None:
        return []

    if len(target_fingerprint) != FINGERPRINT_DIMENSIONS:
        print(f"❌ Target fingerprint wrong size: {len(target_fingerprint)}")
        return []

    library_size = len(filenames)
    max_results = calculate_adaptive_k_linear(library_size)

    print(f"🎯 Library: {library_size} samples → returning {max_results} results (linear scaling)")

    # Get classification for filtering
    target_type = None
    if uploaded_filename and filter_by_type:
        target_type = classify_sample_from_filename(uploaded_filename)
        if target_type != 'unknown':
            print(f"🏷️ Target classified as: {target_type}")

    # Search more candidates than we need for better filtering
    search_k = min(library_size, max(50, max_results * 4))
    target = np_lib.array([target_fingerprint]).astype('float32')

    distances, indices = faiss_index.search(target, search_k)

    # Process results with type filtering
    results = []
    same_type_results = []
    other_type_results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        file_path = filenames[idx]
        filename = os.path.basename(file_path)

        # Skip self-matches
        if uploaded_filename and filename == uploaded_filename:
            continue

        result_item = (file_path, float(distance))

        # Separate by type for better filtering
        if filter_by_type and target_type != 'unknown':
            file_type = classify_sample_from_filename(filename)
            if file_type == target_type:
                same_type_results.append(result_item)
            elif file_type == 'unknown':
                other_type_results.append(result_item)
            # Skip clearly different types (e.g., kick vs hihat)
        else:
            results.append(result_item)

    # Smart result selection
    if filter_by_type and target_type != 'unknown':
        # Prioritize same type, fill with unknowns if needed
        final_results = same_type_results[:max_results]

        if len(final_results) < max_results:
            remaining_slots = max_results - len(final_results)
            final_results.extend(other_type_results[:remaining_slots])

        results = final_results
        same_type_count = len(same_type_results)
    else:
        results = results[:max_results]
        same_type_count = 0

    print(f"✅ Linear adaptive search: {len(results)} results")
    if same_type_count > 0:
        print(f"   📊 {same_type_count} same-type, {len(results) - same_type_count} other/unknown")

    # Fallback: if we have very few same-type results, search all types
    if len(results) < 3 and filter_by_type and target_type != 'unknown':
        print("🔄 Too few same-type results - including all types")
        return find_similar_faiss_linear_adaptive(
            target_fingerprint, faiss_index, filenames, uploaded_filename, False
        )

    return results


def preview_scaling():
    """Preview how the scaling works for different library sizes"""
    test_sizes = [25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500]

    print("📊 Linear Adaptive Scaling Preview:")
    print("Library Size → Results Returned")
    print("-" * 35)

    for size in test_sizes:
        results = calculate_adaptive_k_linear(size)
        percentage = (results / size) * 100
        print(f"{size:4d} samples → {results:2d} results ({percentage:.1f}% of library)")

    print("\n🎯 Your 300 samples →", calculate_adaptive_k_linear(300), "results")
    return True

# Keep all your existing utility functions unchanged
def classify_sample_from_filename(filename):
    """Extract sample type from filename keywords"""
    filename_lower = filename.lower()
    filename_clean = os.path.splitext(filename_lower)[0]

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

    for sample_type, keywords in sample_types.items():
        for keyword in keywords:
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

    bpm_patterns = [
        r'(\d{2,3})bpm',
        r'(\d{2,3})_bpm',
        r'(\d{2,3}) bpm',
        r'bpm(\d{2,3})',
        r'(\d{2,3})$'
    ]

    filename_lower = filename.lower()
    for pattern in bpm_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            bpm = int(match.group(1))
            if 60 <= bpm <= 200:
                return bpm

    return None


def extract_key_from_filename(filename):
    """Extract musical key from filename if present"""
    import re

    key_pattern = r'([A-G][#b]?[m]?)'
    matches = re.findall(key_pattern, filename)

    if matches:
        return matches[0]

    return None


def save_faiss_system(fingerprint_db, faiss_index, filenames, db_path="fingerprint_db.pkl",
                      index_path="faiss_index.bin", names_path="filenames.pkl"):
    """Save the complete fast search system"""
    faiss_lib = _require_module(faiss, 'faiss')

    with open(db_path, "wb") as f:
        pickle.dump(fingerprint_db, f)

    faiss_lib.write_index(faiss_index, index_path)

    with open(names_path, "wb") as f:
        pickle.dump(filenames, f)

    print(f"Enhanced search system saved!")


def load_faiss_system(db_path="fingerprint_db.pkl", index_path="faiss_index.bin", names_path="filenames.pkl"):
    """Load the complete fast search system"""
    faiss_lib = _require_module(faiss, 'faiss')

    fingerprint_db = None
    faiss_index = None
    filenames = None

    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            fingerprint_db = pickle.load(f)
        print(f"Loaded fingerprint database from {db_path}")

    if os.path.exists(index_path):
        faiss_index = faiss_lib.read_index(index_path)
        print(f"Loaded FAISS index from {index_path}")

    if os.path.exists(names_path):
        with open(names_path, "rb") as f:
            filenames = pickle.load(f)
        print(f"Loaded filenames from {names_path}")

    return fingerprint_db, faiss_index, filenames


def get_audio_files(folder_path):
    """Get all audio files recursively"""
    audio_files = []
    supported_formats = {'.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a'}

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                audio_files.append(os.path.join(root, filename))

    return audio_files


def build_fingerprint_database(folder_path):
    """Build fingerprint database with enhanced features"""
    audio_files = get_audio_files(folder_path)
    fingerprint_database = {}

    print(f"Processing {len(audio_files)} audio files with enhanced algorithm...")

    for i, file_path in enumerate(audio_files):
        print(f"Processing file {i + 1}/{len(audio_files)}: {os.path.basename(file_path)}")

        try:
            fingerprint = generate_fingerprint(file_path)
            fingerprint_database[file_path] = fingerprint

            if i % 10 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            fingerprint_database[file_path] = None

    gc.collect()

    valid_db = {k: v for k, v in fingerprint_database.items() if v is not None}
    print(f"Successfully processed {len(valid_db)} out of {len(audio_files)} files with enhanced features")

    return valid_db


def update_fingerprint_database_incremental(fingerprint_db, samples_folder):
    """Only process new files, keep existing fingerprints"""
    current_files = set(get_audio_files(samples_folder))

    if fingerprint_db is None:
        fingerprint_db = {}

    existing_files = set(fingerprint_db.keys())
    new_files = current_files - existing_files
    removed_files = existing_files - current_files

    # Remove deleted files
    for removed_file in removed_files:
        del fingerprint_db[removed_file]
        print(f"🗑️ Removed: {os.path.basename(removed_file)}")

    # Process new files
    if new_files:
        print(f"🆕 Processing {len(new_files)} new samples with enhanced algorithm...")

        for file_path in new_files:
            print(f"  Processing NEW: {os.path.basename(file_path)}")
            fingerprint = generate_fingerprint(file_path)
            fingerprint_db[file_path] = fingerprint

    changes_made = len(new_files) > 0 or len(removed_files) > 0

    if changes_made:
        print(f"✅ Enhanced database updated: +{len(new_files)} files, -{len(removed_files)} files")
    else:
        print("✅ No changes detected")

    return fingerprint_db, changes_made


def ensure_library_is_current(fingerprint_db, faiss_index, filenames, samples_folder, db_path=None, index_path=None,
                              names_path=None):
    """Check for library changes and rebuild the enhanced index if needed"""
    fingerprint_db, changes_made = update_fingerprint_database_incremental(fingerprint_db, samples_folder)

    if changes_made:
        print("📁 Library updated - rebuilding enhanced search index...")
        faiss_index, filenames = build_faiss_index(fingerprint_db)

        if db_path and index_path and names_path:
            save_faiss_system(fingerprint_db, faiss_index, filenames, db_path, index_path, names_path)
        else:
            save_faiss_system(fingerprint_db, faiss_index, filenames)

        print("✅ Enhanced search index updated!")

    return fingerprint_db, faiss_index, filenames, changes_made


def get_library_stats(fingerprint_db):
    """Get statistics about the sample library"""
    if not fingerprint_db:
        return {}

    total_samples = len(fingerprint_db)

    type_counts = {}
    for file_path in fingerprint_db.keys():
        filename = os.path.basename(file_path)
        sample_type = classify_sample_from_filename(filename)
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

    # Removed BPM ranges - not useful without tempo detection
    return {
        'total_samples': total_samples,
        'types': type_counts
    }


def generate_fingerprint_with_fallback(file_path):
    """Try enhanced algorithm, fall back to basic if it fails"""
    try:
        # Try enhanced 12-feature algorithm first
        return generate_fingerprint(file_path)
    except Exception as e:
        print(f"⚠️ Enhanced algorithm failed for {file_path}, trying basic fallback: {e}")
        try:
            # Fallback to basic 6-feature algorithm, padded to 12
            return generate_basic_fingerprint_padded(file_path)
        except Exception as e2:
            print(f"❌ Both algorithms failed for {file_path}: {e2}")
            return np.zeros(FINGERPRINT_DIMENSIONS, dtype='float32')


def generate_basic_fingerprint_padded(file_path):
    """Basic 6-feature algorithm padded to 12 features"""
    librosa_lib = _require_module(librosa, 'librosa')
    np_lib = _require_module(np, 'numpy')

    y, sr = librosa_lib.load(file_path, sr=8000, duration=2, mono=True)
    if len(y) == 0:
        return np_lib.zeros(FINGERPRINT_DIMENSIONS, dtype='float32')

    y = librosa_lib.util.normalize(y)

    # Basic features
    mfccs = librosa_lib.feature.mfcc(y=y, sr=sr, n_mfcc=3, n_fft=256, hop_length=256)
    mfcc_mean = np_lib.mean(mfccs, axis=1)

    spectral_centroid = np_lib.mean(librosa_lib.feature.spectral_centroid(y=y, sr=sr))
    rms_energy = np_lib.mean(librosa_lib.feature.rms(y=y))
    zcr = np_lib.mean(librosa_lib.feature.zero_crossing_rate(y))

    # Create 12-feature vector (6 real + 10 padding)
    features = np_lib.zeros(FINGERPRINT_DIMENSIONS, dtype='float32')
    features[0] = float(mfcc_mean[0])
    features[1] = float(mfcc_mean[1])
    features[2] = float(mfcc_mean[2])
    features[3] = float(spectral_centroid)
    features[4] = float(rms_energy)
    features[5] = float(zcr)
    # features[6:12] remain zeros (padding)

    return features