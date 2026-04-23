from flask import Flask, render_template, request, session, g
import numpy as np
import sqlite3
import re
import pandas as pd
import librosa
import soundfile as sf
import gc
import time
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
logging.basicConfig(filename='app.log', level=logging.ERROR)
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    """Decorator to require login for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('Loggedin'):
            return redirect('/login.html')
        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__)
app.secret_key = "KjhLJF54f6ds234H"

DATABASE = "mydb.sqlite3"

audio_dir = 'audio_files'

dataset = pd.read_csv('dataset.csv')

num_mfcc = 100
num_mels = 128
num_chroma = 50

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

@app.route('/')
def home():
    background_image = "/static/image1.jpg"
    return render_template('index.html', background_image=background_image)

@app.route('/login.html', methods=['GET', 'POST'])
def login():

    background_image = "/static/image5.jpg"
    if request.method == 'POST':
        email = request.form["email"]
        password = request.form["password"]
        try:
            cursor = get_db().cursor()
            cursor.execute("SELECT * FROM REGISTER WHERE EMAIL = ?", (email,))
            account = cursor.fetchone()
            if account and check_password_hash(account[3], password):
                session['Loggedin'] = True
                session['id'] = account[1]
                session['email'] = account[1]
                return render_template('model.html', background_image=background_image)
            else:
                msg = "Incorrect Email/password"
                return render_template('login.html', msg=msg, background_image=background_image)
        except Exception as e:
            logging.error(f"Login DB error: {e}")
            msg = "Login failed. Please try again."
            return render_template('login.html', msg=msg, background_image=background_image)
    else:
        return render_template('login.html',background_image=background_image)

@app.route('/contact.html')
def contact():
    background_image = "/static/image3.jpg"
    return render_template('contact.html', background_image=background_image)

@app.route('/about.html')
def about():
    background_image = "/static/image2.jpg"
    return render_template('about.html', background_image=background_image)

@app.route('/index.html')
def home1():
    background_image = "/static/image1.jpg"
    return render_template('index.html', background_image=background_image)

@app.route('/chart.html')
def chart():
    return render_template('chart.html')


@app.route('/register.html', methods=['GET', 'POST'])
def signup():
    msg = ''
    background_image = "/static/image4.jpg"

    if request.method == 'POST':
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm-password"]
        try:
            cursor = get_db().cursor()
            cursor.execute("SELECT * FROM REGISTER WHERE username = ?", (username,))
            account_username = cursor.fetchone()
            cursor.execute("SELECT * FROM REGISTER WHERE email = ?", (email,))
            account_email = cursor.fetchone()
            if account_username:
                msg = "Username already exists"
            elif account_email:
                msg = "Email already exists"
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                msg = "Invalid Email Address!"
            elif password != confirm_password:
                msg = "Passwords do not match!"
            else:
                hashed_pw = generate_password_hash(password)
                cursor.execute("INSERT INTO REGISTER (username, email, password) VALUES (?,?,?)",
                               (username, email, hashed_pw))
                get_db().commit()
                msg = "You have successfully registered"
        except Exception as e:
            logging.error(f"Registration DB error: {e}")
            msg = "Registration failed. Please try again."

    return render_template('register.html', msg=msg, background_image=background_image)


@app.route('/model.html', methods=['GET', 'POST'])
@login_required
def model():
    """Handle audio upload and prediction."""
    background_image = "/static/image5.jpg"
    loader_visible = False
    if request.method == 'POST':
        selected_file = request.files['audio_file']
        if not allowed_file(selected_file.filename):
            msg = "Only .wav and .mp3 files are allowed."
            return render_template('model.html', msg=msg, background_image=background_image, loader_visible=loader_visible)
        filename = secure_filename(selected_file.filename)
        file_path = os.path.join(audio_dir, filename)
        selected_file.save(file_path)
        loader_visible = True
        try:
            file_name = os.path.basename(file_path)
            X, sample_rate = sf.read(file_path)
            if X.ndim > 1:
                X = np.mean(X, axis=1)  # Convert to mono if stereo
            target_sr = 22050
            if sample_rate != target_sr:
                X = librosa.resample(X, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
            chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
            flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
            features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))
            distances = np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1)
            closest_match_idx = np.argmin(distances)
            closest_match_label = dataset.iloc[closest_match_idx, -1]
            total_distance = np.sum(distances)
            closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
            closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)
            if closest_match_label == 'deepfake':
                file_label = f"File: {file_name}"
                result_label = f"Result: Fake"
            else:
                file_label = f"File: {file_name}"
                result_label = f"Result: Real"
            # Force garbage collection to ensure file handles are released
            gc.collect()
            for _ in range(5):
                try:
                    os.remove(file_path)
                    break
                except PermissionError:
                    time.sleep(0.5)
            else:
                logging.warning(f"Could not delete file {file_path} after several attempts.")
            return render_template('model.html', file_label=file_label, result_label=result_label, confidence=closest_match_prob_percentage, background_image=background_image, loader_visible=loader_visible)
        except Exception as e:
            logging.error(f"Audio processing error: {e}")
            msg = "Error processing audio file."
            return render_template('model.html', msg=msg, background_image=background_image, loader_visible=loader_visible)
    else:
        return render_template('model.html', background_image=background_image, loader_visible=loader_visible)


if __name__ == "__main__":
    app.run(debug=True)

