import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Core
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ai-attendance-system-super-secret-key-2023'

    # DB (kept for future SQLAlchemy migration)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'database', 'attendance.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File uploads
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    UPLOAD_FOLDER = os.path.abspath(os.path.join(basedir, 'dataset'))  # absolute path
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

    # Face recognition settings (match keys used in app.py)
    FACE_RECOGNITION_MODEL = os.environ.get('FACE_RECOGNITION_MODEL', 'hog')  # 'hog' or 'cnn'
    FACE_RECOGNITION_TOLERANCE = float(os.environ.get('FACE_RECOGNITION_TOLERANCE', 0.6))
    FACE_RECOGNITION_NUM_JITTERS = int(os.environ.get('FACE_RECOGNITION_NUM_JITTERS', 1))

    # App defaults (align with settings table defaults)
    ATTENDANCE_THRESHOLD = 0.7
    AUTO_RETRAIN_MODEL = True
    MAX_IMAGE_SIZE = 1024
    DEFAULT_DEPARTMENT = 'Computer Science'

    # Sessions
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = bool(int(os.environ.get('SESSION_COOKIE_SECURE', '0')))
    REMEMBER_COOKIE_SECURE = bool(int(os.environ.get('REMEMBER_COOKIE_SECURE', '0')))
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True

    # API
    API_PREFIX = '/api/v1'
    API_THROTTLING = True
    API_RATE_LIMIT = '100 per day'

    # Mail (optional)
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@ai-attendance.com')

    # JSON/Logging
    JSONIFY_PRETTYPRINT_REGULAR = False
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Exports/Backups
    EXPORT_FORMATS = ['csv', 'json', 'pdf']
    EXPORT_DEFAULT_FORMAT = 'csv'
    AUTO_BACKUP = True
    BACKUP_INTERVAL_HOURS = 24
    MAX_BACKUP_FILES = 7

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    ENV = 'development'
    CORS_ORIGINS = ['http://localhost:5000', 'http://127.0.0.1:5000']
    TRAP_HTTP_EXCEPTIONS = True
    TRAP_BAD_REQUEST_ERRORS = True

class TestingConfig(Config):
    DEBUG = False
    TESTING = True
    ENV = 'testing'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    ENV = 'production'
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    LOG_LEVEL = 'WARNING'
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else []

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    env = os.environ.get('FLASK_ENV') or 'development'
    return config.get(env, config['default'])

def init_directories():
    directories = [
        os.path.join(basedir, 'database'),
        os.path.join(basedir, 'dataset'),
        os.path.join(basedir, 'encodings'),
        os.path.join(basedir, 'static', 'temp'),
        os.path.join(basedir, 'static', 'results'),
        os.path.join(basedir, 'backups')
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

init_directories()
