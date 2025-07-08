import importlib.util
import sys
import types
from pathlib import Path


def load_audio_processing():
    # Stub heavy optional dependencies
    stub_names = [
        'librosa', 'faiss', 'numpy',
        'scipy', 'scipy.spatial', 'scipy.spatial.distance'
    ]
    for name in stub_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # ensure nested modules exist
    sys.modules['scipy'].spatial = sys.modules['scipy.spatial']
    sys.modules['scipy.spatial'].distance = sys.modules['scipy.spatial.distance']
    sys.modules['scipy.spatial.distance'].cosine = lambda x, y: 0

    path = Path(__file__).resolve().parents[1] / 'utils' / 'audio_processing.py'
    spec = importlib.util.spec_from_file_location('audio_processing', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['audio_processing'] = module
    spec.loader.exec_module(module)
    return module


audio_processing = load_audio_processing()


def test_classify_sample_from_filename():
    classify = audio_processing.classify_sample_from_filename
    assert classify('cool_kick_loop.wav') == 'kick'
    assert classify('snare_clap.wav') == 'snare'
    assert classify('808_bass.wav') == 'bass'
    assert classify('nice_pad_strings.wav') == 'pad'
    assert classify('lead_vocal.wav') == 'lead'
    assert classify('unknown_sample.wav') == 'unknown'


def test_extract_bpm_from_filename():
    bpm = audio_processing.extract_bpm_from_filename
    assert bpm('kick_140bpm.wav') == 140
    assert bpm('track-90_bpm.wav') == 90
    assert bpm('sample_172 bpm.wav') == 172
    assert bpm('bpm128.wav') == 128
    assert bpm('cool_50bpm.wav') is None


def test_extract_key_from_filename():
    key = audio_processing.extract_key_from_filename
    assert key('sample_Cm.wav') == 'Cm'
    assert key('beat_in_F#_120bpm.wav') == 'F#'
    assert key('guitar-Bb-solo.wav') == 'Bb'
    assert key('no_key_here.wav') is None
