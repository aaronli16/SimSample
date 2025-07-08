# Splice Finder

Splice Finder is a small demo project that lets you search your own audio sample library for files that sound similar to a reference clip.  The backend is a Flask app that fingerprints every sample you upload and builds a FAISS index for fast similarity search.  The frontend is a lightweight React interface served from the Flask server.

(I got lazy digging through hundreds of samples trying to find a sound similar to my favorite song...)
## Setup

### 1. Install Python dependencies

The repository provides two requirement files—one for macOS and one for Windows PCs.  Choose the one that matches your system:

```bash
# On macOS
pip install -r requirementsMAC.txt

# On Windows
pip install -r requirementsPC.txt
```

`ffmpeg` must be installed and available on your `PATH` for the `pydub` library to process audio files.

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your Google OAuth credentials and a
random Flask `SECRET_KEY`:

```bash
cp .env.example .env
# then edit .env
```

### 3. Start the application

Run the Flask server from the repository root:

```bash
python main.py
```

The app listens on port `5050`.  Open `http://localhost:5050` in your browser to load the React interface.

### Docker (optional)

You can also run Splice Finder in a container. Build the image and start the app
like this:

```bash
docker build -t splice-finder .
docker run --env-file .env -p 5050:5050 splice-finder
```

The container exposes port `5050`, so browse to `http://localhost:5050` once it
starts.


### Deployment

For production environments use a WSGI server such as **Gunicorn**:

```bash
gunicorn -b 0.0.0.0:5050 main:app
```


Gunicorn works on Linux and macOS. On Windows you can instead install **waitress** and run:
```bash
waitress-serve --port 5050 main:app
```

This command is what the provided Dockerfile runs by default.

The server also exposes a simple health check at `/health` which returns `OK`
when the app is running. This can be used for monitoring in production.

## Usage

1. Upload a folder of audio samples (either as a zip file or by selecting a folder).  The server builds a fingerprint database and FAISS index.
2. Once the library is built, upload a single sample to search for similar sounds.
3. The interface lists the best matches with inline audio players and lets you open the file location in your system’s file explorer.
4. You can clear your uploaded library at any time.

## Features

- Builds a personal sample library from uploaded files.
- Fast similarity search using FAISS.
- Basic classification by filename keywords (kick, snare, etc.).
- Displays BPM and key if present in the filename.
- React interface with audio players and an option to reveal matched files in Finder/Explorer.

## Prerequisites

- Python 3.9 or newer
- `ffmpeg` installed and accessible in your shell

This project is meant as a proof of concept and ships with a small set of example samples in `audio_samples/`.
## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to propose changes.



## Setup Instructions

### Environment Variables
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
2. Fill in your actual values in .env:

SECRET_KEY: Generate a secure random key using python -c "import secrets; print(secrets.token_urlsafe(32))"
GOOGLE_CLIENT_ID & GOOGLE_CLIENT_SECRET: Get these from Google Cloud Console




3. Never commit .env to Git - it contains sensitive secrets
## License

This project is licensed under the [MIT License](LICENSE).
