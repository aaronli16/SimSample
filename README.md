# 🎵 SimSample

Find similar sounds in your sample library instantly. Upload your audio samples and discover musically similar tracks using AI-powered audio fingerprinting.
## 🧠 The Story

Built by a music producer who got tired of manually digging through sample folders looking for "that one kick that sounds like the one in my reference track." 

Started as a simple script, evolved into a full-scale production system that handles the real-world challenges of audio processing at scale - memory constraints, large file uploads, and the need for actually usable search results.

**🚀 [Try the Live Demo](https://simsample-371783151021.us-central1.run.app/)**


## What It Does

As a music producer, manually browsing through hundreds of samples to find a specific sound is time-consuming and frustrating. SimSample solves this by letting you:

1. **Upload your sample library** (drag & drop folders, any size)
2. **Drop in a reference track** (the sound you're looking for) 
3. **Get instant results** (similar samples ranked by audio similarity)

Perfect for producers who want to find samples that match the vibe of their favorite tracks.

## 🎯 Key Features

- **🎵 Audio Similarity Search**: Upload a reference sample and find similar sounds in your library
- **📁 Large Library Support**: Handles 500MB+ sample collections automatically
- **⚡ Fast Results**: Sub-second similarity search once processed
- **🏗️ Smart Processing**: Automatically chunks large uploads to handle any library size
- **📱 Clean Interface**: Drag & drop uploads with real-time progress tracking
- **🔄 Auto-Retry**: Handles cloud startup issues gracefully

## 🛠️ Technical Highlights

Built to handle real-world music production workflows with enterprise-grade architecture:

- **Audio Processing**: MFCC feature extraction using librosa for perceptual similarity
- **Search Engine**: FAISS (Facebook AI Similarity Search) for instant vector similarity queries  
- **Scalable Architecture**: Chunked batch processing to overcome cloud memory constraints
- **Production Deployment**: Google Cloud Run with automatic scaling and error handling
- **Frontend**: React with real-time progress tracking and responsive design

### Architecture Overview

```
Sample Upload → Chunked Processing → Audio Fingerprinting → FAISS Index → Instant Search
     ↓              ↓                     ↓                 ↓            ↓
[Large Files] → [Batch System] → [MFCC Features] → [Vector DB] → [Real-time Query]
```

**Technical Challenge Solved**: Processing 400+ audio files (500MB+) in cloud environments with strict memory limits through distributed batch architecture.

## 🎬 How It Works

1. **Upload**: Drag and drop your sample folder - automatically chunks large libraries
2. **Processing**: Extracts audio fingerprints using MFCC analysis in the background  
3. **Search**: Upload any reference track and get similar samples ranked by audio similarity
4. **Discover**: Find samples you forgot you had or discover new creative combinations

## 🎵 Perfect For

- **Music Producers**: Find similar drums, bass lines, or melodic elements
- **Sample Diggers**: Organize and explore large sample collections  
- **Beat Makers**: Discover creative combinations and variations
- **Audio Enthusiasts**: Explore how audio similarity algorithms work

## 🤝 Contributing

Want to improve SimSample? Contributions welcome!

**Areas for improvement:**
- 🎨 UI/UX enhancements (mobile optimization, dark mode)
- ⚡ Performance improvements (faster processing, better algorithms)  
- 🔧 New features (batch download, advanced filters, sample previews)
- 🧠 ML improvements (neural similarity models, better feature extraction)

**How to contribute:**
1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test with the live demo
4. Submit a pull request with a clear description

All contributions are reviewed before merging to protect the live production deployment.

## 📊 Performance Stats

- **Library Size**: Successfully tested with 500MB+ sample collections
- **File Count**: Processes 400+ audio files reliably  
- **Processing Speed**: ~20 files per batch with automatic error recovery
- **Search Speed**: Sub-second similarity queries via FAISS indexing
- **Deployment**: Production-ready on Google Cloud Run with 99.9% uptime

## 🧪 Quick Test

Don't have your own samples ready?

👉 [Download Test Sample Folder: https://drive.google.com/drive/folders/1iCt6Fv-9TC__pAHTsKmfwdrXWYJBKqk0?usp=sharing]
1. Download and unzip the folder.
2. Upload the samples into SimSample using the drag-and-drop interface.
3. Try a similarity search with a wanted sound

Don't have a sound you want to test with?

Test Sound: https://drive.google.com/file/d/18qN46KlzK3BxRfWScY8Aj21UNtotJRip/view?usp=sharin


## 📞 Connect

**Aaron Li** - Music Producer & Developer

- [LinkedIn](https://linkedin.com/in/aaron-li-0b4161248) 
- [Instagram](https://instagram.com/_aaronlii)
- [Portfolio](https://github.com/aaronli16)

---

**🚀 [Try SimSample Now](https://simsample-371783151021.us-central1.run.app/)** - Upload your samples and discover similar sounds instantly.
