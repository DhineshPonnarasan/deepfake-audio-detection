
# Deepfake Audio Detection System

## Key Highlights

- **Secure authentication with password hashing**
- **Protected routes using session-based authentication**
- **Safe file upload handling with validation**
- **Deepfake detection with confidence score output**

## Overview

The Deepfake Audio Detection System is an AI-powered web application designed to detect and prevent malicious use of deepfake audio technology. Leveraging advanced machine learning and signal processing techniques, this system analyzes audio files to distinguish between genuine and manipulated (deepfake) recordings, helping to safeguard the integrity of voice communications.

## Features
- **Deepfake Audio Detection:** Analyze audio files to identify deepfake manipulations.
- **User Authentication:** Secure registration and login system.
- **Audio Upload:** Simple interface for uploading audio files for analysis.
- **Real-Time Feedback:** Immediate results on audio authenticity.
- **Professional UI:** Clean, responsive design using Bootstrap.
- **SQLite Database:** Lightweight, file-based storage for user data.

## Tech Stack
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS (Bootstrap), JavaScript
- **Audio Processing:** librosa, numpy, pandas, soundfile, scikit-learn
- **Database:** SQLite

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DhineshPonnarasan/deepfake-audio-detection
   cd deepfake-audio-detection
   ```
2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python app.py
   ```

## Usage
- Register a new account or log in.
- Upload an audio file (WAV/MP3) for analysis.
- View the detection results and confidence score.

## Screenshots
<!-- Add screenshots of the UI here -->

## Future Improvements
- Add support for more audio formats.
- Improve detection accuracy with larger datasets.
- Integrate REST API for programmatic access.
- Add user roles and admin dashboard.
- Deploy to cloud platforms (Azure, AWS, etc.).

---

© 2026 Deepfake Audio Detection System
