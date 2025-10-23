import os
import cv2
import numpy as np
import pickle
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import face_recognition
import sqlite3
import time
import threading
from queue import Queue
from io import BytesIO
from functools import lru_cache

from config import get_config

app = Flask(__name__)
app.config.from_object(get_config())

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

def zip(*iterables):
    """Python zip function for templates"""
    return list(__builtins__['zip'](*iterables))

# Add this to your app context
@app.context_processor
def utility_processor():
    return dict(zip=zip)

def db_path():
    return os.path.join('database', 'attendance.db')

def init_db():
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect(db_path())
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  roll_number TEXT UNIQUE NOT NULL,
                  email TEXT,
                  department TEXT,
                  image_path TEXT,
                  face_encoding BLOB,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id INTEGER,
                  date TEXT NOT NULL,
                  time TEXT,
                  status TEXT NOT NULL,
                  confidence REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (student_id) REFERENCES students (id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  key TEXT UNIQUE NOT NULL,
                  value TEXT,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    default_settings = [
        ('attendance_threshold', '0.6'),
        ('auto_retrain_model', 'true'),
        ('max_image_size', '1024'),
        ('default_department', 'Computer Science'),
        ('face_recognition_model', app.config.get('FACE_RECOGNITION_MODEL', 'hog')),
        ('face_recognition_tolerance', str(app.config.get('FACE_RECOGNITION_TOLERANCE', 0.5))),  # Improved
        ('face_recognition_num_jitters', str(app.config.get('FACE_RECOGNITION_NUM_JITTERS', 2))),  # Improved
        ('face_detection_upsamples', str(app.config.get('FACE_DETECTION_UPSAMPLES', 2))),  # New
    ]
    for key, value in default_settings:
        c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))

    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_unique ON attendance(student_id, date)")
    conn.commit()
    conn.close()

def ensure_db_initialized():
    """Ensure database is properly initialized"""
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

def get_setting(key, default=None):
    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else default

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def _load_all_students():
    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT id, name, roll_number, image_path, face_encoding FROM students")
    rows = c.fetchall()
    conn.close()
    return rows

# ========== IMPROVED FACE RECOGNITION CLASS ==========
class OptimizedFaceRecognition:
    def __init__(self):
        self.known_encodings = None
        self.known_names = None
        self.known_ids = None
        self.last_loaded = 0
        self.cache_timeout = 300  # 5 minutes
        
    def _load_encodings_cached(self):
        """Cache encodings in memory to avoid disk I/O"""
        current_time = time.time()
        if (self.known_encodings is None or 
            current_time - self.last_loaded > self.cache_timeout):
            
            enc_path = os.path.join("encodings", "encodings.pickle")
            if os.path.exists(enc_path):
                with open(enc_path, "rb") as f:
                    data = pickle.loads(f.read())
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
                self.known_ids = data["ids"]
                self.last_loaded = current_time
                print(f"✅ Loaded {len(self.known_encodings)} encodings into cache")
                print(f"🔍 DEBUG - KNOWN NAMES IN SYSTEM: {self.known_names}")
        
    def process_frame_fast(self, image_path, is_live=False):
        """IMPROVED VERSION: Better recognition with enhanced parameters"""
        start_time = time.time()
        
        # Load cached encodings
        self._load_encodings_cached()
        if not self.known_encodings:
            return {"error": "No students registered yet!"}
        
        # Load and optimize image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Invalid image"}
        
        # Enhanced image preprocessing
        h, w = image.shape[:2]
        max_size = 640 if is_live else 800
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"🔄 Resized image from {w}x{h} to {new_w}x{new_h}")
        
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # IMPROVED Face detection with better parameters
        model = get_setting('face_recognition_model', 'hog')
        upsamples = int(get_setting('face_detection_upsamples', 2))
        
        if is_live:
            locations = face_recognition.face_locations(
                rgb, 
                number_of_times_to_upsample=1,  # Faster for live
                model=model
            )
            num_jitters = 1
        else:
            locations = face_recognition.face_locations(
                rgb, 
                number_of_times_to_upsample=upsamples,  # Better detection
                model=model
            )
            num_jitters = int(float(get_setting('face_recognition_num_jitters', 2)))
        
        print(f"📍 Found {len(locations)} faces in {time.time() - start_time:.2f}s")
        
        if not locations:
            return {
                "error": "No faces detected in the image. Please ensure faces are clearly visible.",
                "total_faces": 0,
                "recognized_count": 0,
                "processing_time": time.time() - start_time
            }
        
        # Get encodings
        encodings = face_recognition.face_encodings(
            rgb, 
            known_face_locations=locations, 
            num_jitters=num_jitters
        )
        
        # IMPROVED RECOGNITION LOGIC
        tolerance = float(get_setting('face_recognition_tolerance', 0.5))
        min_confidence = 0.65  # More reasonable threshold
        
        recognized_names, recognized_ids, confidence_scores = [], [], []
        
        for encoding in encodings:
            if len(self.known_encodings) > 0:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                confidence = 1.0 - best_dist
                
                print(f"🎯 Best match: {self.known_names[best_idx]}, Distance: {best_dist:.4f}, Confidence: {confidence:.1%}")
                
                # IMPROVED: Dynamic threshold based on dataset size
                dynamic_threshold = tolerance
                if len(self.known_encodings) > 30:  # More students = stricter matching
                    dynamic_threshold = max(0.4, tolerance - 0.05)
                
                if best_dist <= dynamic_threshold and confidence >= min_confidence:
                    name = self.known_names[best_idx]
                    student_id = self.known_ids[best_idx]
                    similarity = confidence
                    print(f"✅ CONFIDENT MATCH: {name} (confidence: {confidence:.1%})")
                else:
                    name = "Unknown"
                    student_id = None
                    similarity = confidence
                    if best_dist <= dynamic_threshold:
                        print(f"⚠️ LOW CONFIDENCE: {self.known_names[best_idx]} (confidence: {confidence:.1%} < {min_confidence:.0%})")
                    else:
                        print(f"❌ NO MATCH: Best distance {best_dist:.4f} > threshold {dynamic_threshold}")
            else:
                name = "Unknown"
                student_id = None
                similarity = 0.0
                
            recognized_names.append(name)
            recognized_ids.append(student_id)
            confidence_scores.append(float(similarity))
        
        processing_time = time.time() - start_time
        
        print(f"🎯 FINAL RECOGNITION RESULTS:")
        print(f"   Recognized: {recognized_names}")
        print(f"   Confidences: {confidence_scores}")
        
        # Generate result image
        result_image = None
        if not is_live and len(locations) > 0:
            result_image = self._generate_result_image(image, locations, recognized_names, confidence_scores)
        
        return {
            "result_image": result_image,
            "recognized": recognized_names,
            "recognized_ids": recognized_ids,
            "confidence_scores": confidence_scores,
            "total_faces": len(locations),
            "recognized_count": len([n for n in recognized_names if n != "Unknown"]),
            "processing_time": processing_time
        }

    def _generate_result_image(self, image, locations, names, confidences):
        """Generate annotated result image with better visualization"""
        for ((top, right, bottom, left), name, confidence) in zip(locations, names, confidences):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # Enhanced label with better formatting
            label = f"{name} ({confidence:.1%})" if name != "Unknown" else f"Unknown ({confidence:.1%})"
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        result_path = os.path.join('static', 'results', result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, image)
        
        return f"results/{result_filename}"

# Initialize optimized recognizer
optimized_recognizer = OptimizedFaceRecognition()

# ========== IMPROVED MODEL TRAINING ==========
def train_model():
    """Enhanced model training with better error handling and validation"""
    start_time = time.time()
    
    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT id, name, roll_number, image_path, face_encoding FROM students")
    students = c.fetchall()
    
    known_encodings, known_names, known_ids = [], [], []
    num_jitters = int(float(get_setting('face_recognition_num_jitters', 2)))
    model = get_setting('face_recognition_model', 'hog')
    upsamples = int(get_setting('face_detection_upsamples', 2))
    
    processed = 0
    failed_students = []
    
    for student_id, name, roll_number, image_path, face_encoding_blob in students:
        encoding_found = False
        
        # Try to use cached encoding first
        if face_encoding_blob:
            try:
                face_encoding = np.frombuffer(face_encoding_blob, dtype=np.float64)
                if len(face_encoding) == 128:  # Validate encoding size
                    known_encodings.append(face_encoding)
                    known_names.append(f"{name} ({roll_number})")
                    known_ids.append(student_id)
                    processed += 1
                    encoding_found = True
                    print(f"✅ Using cached encoding for {name}")
                    continue
            except Exception as e:
                print(f"⚠️ Error loading cached encoding for {name}: {e}")
        
        # Generate new encoding from image
        if not encoding_found and image_path and os.path.exists(image_path):
            try:
                image = face_recognition.load_image_file(image_path)
                
                # Enhanced image preprocessing
                h, w = image.shape[:2]
                if max(h, w) > 800:
                    scale = 800 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                
                # Improved face detection with better parameters
                locations = face_recognition.face_locations(
                    image, 
                    number_of_times_to_upsample=upsamples,
                    model=model
                )
                
                if not locations:
                    # Try with different model as fallback
                    alt_model = 'cnn' if model == 'hog' else 'hog'
                    locations = face_recognition.face_locations(
                        image, 
                        number_of_times_to_upsample=upsamples,
                        model=alt_model
                    )
                
                if locations:
                    encodings = face_recognition.face_encodings(
                        image, 
                        known_face_locations=locations, 
                        num_jitters=num_jitters
                    )
                    
                    if encodings:
                        encoding = encodings[0]
                        # Save encoding to database
                        c.execute("UPDATE students SET face_encoding = ? WHERE id = ?", 
                                 (encoding.tobytes(), student_id))
                        known_encodings.append(encoding)
                        known_names.append(f"{name} ({roll_number})")
                        known_ids.append(student_id)
                        processed += 1
                        print(f"✅ Generated new encoding for {name}")
                    else:
                        failed_msg = f"{name} - No encodings generated"
                        failed_students.append(failed_msg)
                        print(f"❌ {failed_msg}")
                else:
                    failed_msg = f"{name} - No face detected in image"
                    failed_students.append(failed_msg)
                    print(f"❌ {failed_msg}")
                    
            except Exception as e:
                error_msg = f"{name} - Processing error: {str(e)}"
                failed_students.append(error_msg)
                print(f"❌ {error_msg}")
        else:
            failed_msg = f"{name} - Image file not found"
            failed_students.append(failed_msg)
            print(f"❌ {failed_msg}")
    
    # Save training data
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "ids": known_ids,
        "last_updated": datetime.now().isoformat(),
        "total_students": len(known_names),
        "failed_students": failed_students,
        "training_time": time.time() - start_time
    }
    
    os.makedirs("encodings", exist_ok=True)
    with open(os.path.join("encodings", "encodings.pickle"), "wb") as f:
        pickle.dump(data, f)
    
    conn.commit()
    conn.close()
    
    # Update cache
    optimized_recognizer._load_encodings_cached()
    
    training_time = time.time() - start_time
    print(f"✅ Model trained in {training_time:.2f}s - {processed}/{len(students)} students processed")
    
    if failed_students:
        print(f"❌ Failed to process {len(failed_students)} students:")
        for failed in failed_students:
            print(f"   - {failed}")
    
    return len(known_names)

# ========== OPTIMIZED PROCESSING FUNCTIONS ==========
def process_attendance_image_fast(image_path):
    try:
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        result = optimized_recognizer.process_frame_fast(image_path, is_live=False)
        
        # Ensure result_image uses correct path
        if 'result_image' in result and result['result_image']:
            result['result_image'] = result['result_image'].replace('\\', '/')
        
        return result
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return {"error": f"Processing failed: {str(e)}"}

def process_live_attendance_fast(image_path):
    return optimized_recognizer.process_frame_fast(image_path, is_live=True)

# ========== OPTIMIZED PROCESSING QUEUE ==========
class OptimizedProcessingQueue:
    def __init__(self):
        self.queue = Queue()
        self.results = {}
        self.threads = []
        
        for i in range(2):  # 2 parallel workers
            thread = threading.Thread(target=self._process_queue, daemon=True, name=f"Worker-{i}")
            thread.start()
            self.threads.append(thread)

    def _process_queue(self):
        while True:
            task_id, image_path, is_live = self.queue.get()
            try:
                if is_live:
                    result = process_live_attendance_fast(image_path)
                else:
                    result = process_attendance_image_fast(image_path)
                self.results[task_id] = result
            except Exception as e:
                print(f"❌ Processing error: {str(e)}")
                self.results[task_id] = {"error": str(e)}
            finally:
                self.queue.task_done()
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass

    def add_task(self, task_id, image_path, is_live=False):
        self.results[task_id] = None
        self.queue.put((task_id, image_path, is_live))

    def get_result(self, task_id):
        return self.results.get(task_id)

# Replace your existing queue
processing_queue = OptimizedProcessingQueue()

def get_counts():
    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM students")
    total_students = c.fetchone()[0] or 0
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Present'", (today,))
    present_today = c.fetchone()[0] or 0
    conn.close()
    return total_students, present_today

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'database',
        'dataset', 
        'encodings',
        'static/temp',
        'static/results',
        'static/uploads'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Ensured directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")

# ========== IMPROVED ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        email = request.form.get('email', '')
        department = request.form.get('department', get_setting('default_department', 'Computer Science'))

        if 'image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = secure_filename(f"{roll_number}_{int(time.time())}.{ext}")

            base = app.config['UPLOAD_FOLDER']
            os.makedirs(base, exist_ok=True)
            image_path = os.path.abspath(os.path.join(base, filename))
            try:
                file.save(image_path)
            except Exception as e:
                flash(f'Failed to save file: {e}', 'error')
                return redirect(request.url)

            try:
                if not os.path.exists(image_path):
                    flash('Saved file not found on disk. Check upload path.', 'error')
                    return redirect(request.url)

                # IMPROVED: Enhanced face detection during registration
                image = face_recognition.load_image_file(image_path)
                upsamples = int(get_setting('face_detection_upsamples', 2))
                
                locations = face_recognition.face_locations(
                    image, 
                    number_of_times_to_upsample=upsamples,
                    model=get_setting('face_recognition_model', 'hog')
                )
                
                if not locations:
                    # Try alternative model
                    alt_model = 'cnn' if get_setting('face_recognition_model', 'hog') == 'hog' else 'hog'
                    locations = face_recognition.face_locations(
                        image, 
                        number_of_times_to_upsample=upsamples,
                        model=alt_model
                    )
                
                if not locations:
                    flash('No face detected in the image. Please upload a clearer image with a visible face.', 'error')
                    os.remove(image_path)
                    return redirect(request.url)

                encodings = face_recognition.face_encodings(
                    image, 
                    known_face_locations=locations,
                    num_jitters=int(float(get_setting('face_recognition_num_jitters', 2)))
                )
                
                if not encodings:
                    flash('Failed to generate face encoding. Please try with a different image.', 'error')
                    os.remove(image_path)
                    return redirect(request.url)

                face_encoding = encodings[0].tobytes()

                conn = sqlite3.connect(db_path())
                c = conn.cursor()
                c.execute("""INSERT INTO students (name, roll_number, email, department, image_path, face_encoding)
                             VALUES (?, ?, ?, ?, ?, ?)""",
                          (name, roll_number, email, department, image_path, face_encoding))
                conn.commit()
                conn.close()

                if get_setting('auto_retrain_model', 'true').lower() == 'true':
                    train_model()
                    flash(f'Student {name} registered successfully and model retrained!', 'success')
                else:
                    flash(f'Student {name} registered successfully!', 'success')

                return redirect(url_for('dashboard'))

            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass
                return redirect(request.url)

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    partial = request.args.get('partial') == 'true'

    conn = sqlite3.connect(db_path())
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM students")
    total_students = c.fetchone()[0] or 0

    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Present'", (today,))
    present_today = c.fetchone()[0] or 0

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Present'", (yesterday,))
    present_yesterday = c.fetchone()[0] or 0

    attendance_change = 0.0
    if present_yesterday > 0:
        attendance_change = ((present_today - present_yesterday) / present_yesterday) * 100.0

    dates, attendance_data = [], []
    for i in range(6, -1, -1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        c.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Present'", (date,))
        count = c.fetchone()[0] or 0
        dates.append(date)
        attendance_data.append(count)

    c.execute('''SELECT s.name, a.date, a.time, a.status 
                 FROM attendance a 
                 JOIN students s ON a.student_id = s.id 
                 ORDER BY a.created_at DESC LIMIT 10''')
    recent_attendance = c.fetchall()

    c.execute("SELECT department, COUNT(*) FROM students GROUP BY department")
    department_stats = c.fetchall()

    conn.close()

    attendance_rate = (present_today / total_students * 100.0) if total_students > 0 else 0.0
    absence_rate = (((total_students - present_today) / total_students) * 100.0) if total_students > 0 else 0.0

    if partial:
        return jsonify({
            'success': True,
            'total_students': total_students,
            'present_today': present_today,
            'attendance_rate': round(attendance_rate, 1),
            'absence_rate': round(absence_rate, 1),
            'dates': dates,
            'attendance_data': attendance_data
        })

    return render_template(
        'dashboard.html',
        total_students=total_students,
        present_today=present_today,
        attendance_change=attendance_change,
        dates=dates,
        attendance_data=attendance_data,
        recent_attendance=recent_attendance,
        department_stats=department_stats,
        attendance_rate=round(attendance_rate, 1),
        absence_rate=round(absence_rate, 1)
    )

# ========== IMPROVED MARK ATTENDANCE ROUTE ==========
@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    if request.method == 'GET':
        total_students, present_today = get_counts()
        recent_sessions = []
        return render_template('attendance.html',
                               total_students=total_students,
                               present_today=present_today,
                               recent_sessions=recent_sessions)
    
    elif request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({
                    'status': 'error', 
                    'message': 'No file selected'
                }), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'status': 'error', 
                    'message': 'No file selected'
                }), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                temp_path = os.path.join('static', 'temp', filename)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                file.save(temp_path)

                # Process the image
                result = process_attendance_image_fast(temp_path)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                if 'error' in result:
                    return jsonify({
                        'status': 'error',
                        'message': result['error']
                    }), 500
                
                # Update attendance in database
                conn = sqlite3.connect(db_path())
                c = conn.cursor()
                today = datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.now().strftime("%H:%M:%S")

                # Mark all students as absent first
                c.execute("SELECT id FROM students")
                all_students = c.fetchall()
                for (sid,) in all_students:
                    c.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?", (sid, today))
                    if (c.fetchone()[0] or 0) == 0:
                        c.execute("""INSERT INTO attendance (student_id, date, time, status, confidence)
                                     VALUES (?, ?, ?, ?, ?)""", (sid, today, current_time, 'Absent', 0.0))

                # Update recognized students as present
                recognized_count = 0
                for student_id, similarity in zip(result.get('recognized_ids', []), result.get('confidence_scores', [])):
                    if student_id:
                        c.execute("""UPDATE attendance 
                                     SET status = 'Present', time = ?, confidence = ?
                                     WHERE student_id = ? AND date = ?""",
                                  (current_time, float(similarity), int(student_id), today))
                        recognized_count += 1

                conn.commit()
                conn.close()

                # Fix result_image path
                result_image = result.get('result_image', '')
                if result_image:
                    result_image = result_image.replace('\\', '/')

                # Render the result template directly
                return render_template('attendance_result.html',
                    result_image=result_image,
                    recognized=result.get('recognized', []),
                    confidence_scores=result.get('confidence_scores', []),
                    total_faces=result.get('total_faces', 0),
                    recognized_count=result.get('recognized_count', 0),
                    processing_time=round(result.get('processing_time', 0), 2)
                )
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid file type'
                }), 400
                
        except Exception as e:
            print(f"❌ Error in mark_attendance: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to process attendance: {str(e)}'
            }), 500

@app.route('/attendance_result')
def attendance_result():
    result_image = request.args.get('result_image', '')
    recognized = request.args.getlist('recognized')
    confidence_scores = [float(score) for score in request.args.getlist('confidence_scores')]
    total_faces = int(request.args.get('total_faces', 0))
    recognized_count = int(request.args.get('recognized_count', 0))
    processing_time = float(request.args.get('processing_time', 0))
    
    return render_template('attendance_result.html',
                           result_image=result_image,
                           recognized=recognized,
                           confidence_scores=confidence_scores,
                           total_faces=total_faces,
                           recognized_count=recognized_count,
                           processing_time=processing_time)

@app.route('/check_attendance_result/<task_id>')
def check_attendance_result(task_id):
    result = processing_queue.get_result(task_id)

    if result is None:
        return render_template('processing.html', task_id=task_id)

    if 'error' in result:
        flash(result['error'], 'error')
        return redirect(url_for('mark_attendance'))

    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    c.execute("SELECT id FROM students")
    all_students = c.fetchall()
    for (sid,) in all_students:
        c.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?", (sid, today))
        if (c.fetchone()[0] or 0) == 0:
            c.execute("""INSERT INTO attendance (student_id, date, time, status, confidence)
                         VALUES (?, ?, ?, ?, ?)""", (sid, today, current_time, 'Absent', 0.0))

    for student_id, similarity in zip(result.get('recognized_ids', []), result.get('confidence_scores', [])):
        if student_id:
            c.execute("""UPDATE attendance 
                         SET status = 'Present', time = ?, confidence = ?
                         WHERE student_id = ? AND date = ?""",
                      (current_time, float(similarity), int(student_id), today))

    conn.commit()
    conn.close()

    return render_template('attendance_result.html',
                           result_image=result['result_image'],
                           recognized=result['recognized'],
                           confidence_scores=result['confidence_scores'],
                           total_faces=result['total_faces'],
                           recognized_count=result['recognized_count'])

@app.route('/reports')
def reports():
    date_filter = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    department_filter = request.args.get('department', 'all')

    conn = sqlite3.connect(db_path())
    c = conn.cursor()

    c.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
    dates = [row[0] for row in c.fetchall()]

    c.execute("SELECT DISTINCT department FROM students")
    departments = [row[0] for row in c.fetchall()]

    if department_filter == 'all':
        c.execute('''SELECT s.name, s.roll_number, s.department, a.status, a.time, a.confidence
                     FROM students s 
                     LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ?
                     ORDER BY s.roll_number''', (date_filter,))
    else:
        c.execute('''SELECT s.name, s.roll_number, s.department, a.status, a.time, a.confidence
                     FROM students s 
                     LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ?
                     WHERE s.department = ?
                     ORDER BY s.roll_number''', (date_filter, department_filter))

    attendance_data = c.fetchall()
    total_students = len(attendance_data)
    present_count = sum(1 for r in attendance_data if r[3] == 'Present')
    absent_count = total_students - present_count
    attendance_rate = (present_count / total_students * 100) if total_students > 0 else 0

    conn.close()

    return render_template('reports.html',
                           dates=dates,
                           departments=departments,
                           selected_date=date_filter,
                           selected_department=department_filter,
                           attendance_data=attendance_data,
                           total_students=total_students,
                           present_count=present_count,
                           absent_count=absent_count,
                           attendance_rate=attendance_rate)

@app.route('/export_csv')
def export_csv():
    date_filter = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    department_filter = request.args.get('department', 'all')

    conn = sqlite3.connect(db_path())
    c = conn.cursor()

    if department_filter == 'all':
        c.execute('''SELECT s.name, s.roll_number, s.department, a.status, a.time, a.confidence
                     FROM students s 
                     LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ?
                     ORDER BY s.roll_number''', (date_filter,))
    else:
        c.execute('''SELECT s.name, s.roll_number, s.department, a.status, a.time, a.confidence
                     FROM students s 
                     LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ?
                     WHERE s.department = ?
                     ORDER BY s.roll_number''', (date_filter, department_filter))

    data = c.fetchall()
    conn.close()

    lines = ["Name,Roll Number,Department,Status,Time,Confidence,Date"]
    for row in data:
        lines.append(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5] or 0},{date_filter}")
    csv_text = "\n".join(lines) + "\n"

    mem = BytesIO(csv_text.encode('utf-8'))
    mem.seek(0)

    return send_file(
        mem,
        mimetype='text/csv; charset=utf-8',
        as_attachment=True,
        download_name=f'attendance_{date_filter}_{department_filter}.csv'
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        conn = sqlite3.connect(db_path())
        c = conn.cursor()
        for key in request.form:
            if key != 'retrain_model':
                c.execute("UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?", (request.form[key], key))
        if request.form.get('retrain_model') == 'on':
            train_model()
            flash('Model retrained successfully!', 'success')
        conn.commit()
        conn.close()
        flash('Settings updated successfully!', 'success')
        return redirect(url_for('settings'))

    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT key, value FROM settings")
    settings_data = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return render_template('settings.html', settings=settings_data)

# ========== ALL YOUR EXISTING ROUTES PRESERVED ==========
@app.route('/profile')
def profile():
    """User profile page"""
    return render_template('profile.html')

@app.route('/change_password')
def change_password():
    """Change password page"""
    return render_template('change_password.html')

@app.route('/logout')
def logout():
    """Logout user"""
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/help')
def help():
    """Help and documentation page"""
    return render_template('help.html')

@app.route('/documentation')
def documentation():
    """Documentation page"""
    return render_template('documentation.html')

@app.route('/tutorials')
def tutorials():
    """Tutorials page"""
    return render_template('tutorials.html')

@app.route('/api_reference')
def api_reference():
    """API Reference page"""
    return render_template('api_reference.html')

@app.route('/api/attendance/update', methods=['POST'])
def api_update_attendance():
    payload = request.get_json(silent=True) or {}
    roll = payload.get('roll_number')
    date = payload.get('date') or datetime.now().strftime("%Y-%m-%d")
    status = payload.get('status')
    confidence = float(payload.get('confidence') or 0)

    if status not in ('Present', 'Absent') or not roll:
        return jsonify({'success': False, 'error': 'Invalid input'}), 400

    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT id FROM students WHERE roll_number = ?", (roll,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'success': False, 'error': 'Student not found'}), 404

    sid = row[0]
    c.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?", (sid, date))
    exists = (c.fetchone()[0] or 0) > 0
    now_time = datetime.now().strftime("%H:%M:%S")
    if exists:
        c.execute("""UPDATE attendance SET status = ?, time = ?, confidence = ? 
                     WHERE student_id = ? AND date = ?""",
                  (status, now_time, confidence, sid, date))
    else:
        c.execute("""INSERT INTO attendance (student_id, date, time, status, confidence)
                     VALUES (?, ?, ?, ?, ?)""", (sid, date, now_time, status, confidence))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'status': status, 'time': now_time})

@app.route('/api/students', methods=['GET'])
def api_get_students():
    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT id, name, roll_number, department FROM students")
    students = [{'id': row[0], 'name': row[1], 'roll_number': row[2], 'department': row[3]} for row in c.fetchall()]
    conn.close()
    return jsonify(students)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'students_count': get_students_count(),
        'model_trained': os.path.exists(os.path.join("encodings", "encodings.pickle")),
        'version': '2.1.0'
    })

def get_students_count():
    conn = sqlite3.connect(db_path())
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM students")
    count = c.fetchone()[0] or 0
    conn.close()
    return count

@app.route('/process_live_frame', methods=['POST'])
def process_live_frame():
    """Process live camera frames for real-time recognition"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(f"live_temp_{int(time.time() * 1000)}.jpg")
        temp_path = os.path.join('static', 'temp', filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)

        try:
            # Process with live mode enabled for faster recognition
            result = process_live_attendance_fast(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return jsonify(result)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize everything
    ensure_directories()
    ensure_db_initialized()
    
    # Ensure upload directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('encodings', exist_ok=True)
    os.makedirs(os.path.join('static', 'temp'), exist_ok=True)
    os.makedirs(os.path.join('static', 'results'), exist_ok=True)

    # Train model if students exist
    if get_students_count() > 0:
        print("🔄 Training enhanced face recognition model...")
        train_model()
    else:
        print("ℹ️ No students registered yet. Model training skipped.")

    print("🚀 Starting IMPROVED AI Classroom Attendance System...")
    print(f"📊 Database: {db_path()}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)