import os
import subprocess
import sys

print("🔍 UNIVERSAL PROJECT HEALTH CHECK")
print("=" * 50)

# 1. Check folder structure
print("📁 Checking project structure...")
folders = ['templates', 'static', 'database', 'encodings']
for folder in folders:
    if os.path.exists(folder):
        print(f"✅ Folder exists: {folder}/")
    else:
        print(f"❌ Missing folder: {folder}/")

# 2. Check critical files
print("\n📄 Checking critical files...")
files_to_check = [
    'app.py',
    'templates/base.html',
    'templates/dashboard.html', 
    'templates/register.html',
    'templates/attendance.html',
    'templates/attendance_result.html'
]

for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} ({size} bytes)")
    else:
        print(f"❌ {file} - MISSING!")

# 3. Check if Python can run
print("\n🐍 Checking Python environment...")
try:
    import flask
    print("✅ Flask is installed")
except ImportError:
    print("❌ Flask is NOT installed")

try:
    import face_recognition
    print("✅ face_recognition is installed")
except ImportError:
    print("❌ face_recognition is NOT installed")

try:
    import cv2
    print("✅ OpenCV is installed")
except ImportError:
    print("❌ OpenCV is NOT installed")

# 4. Try to find and test the main app
print("\n🚀 Looking for main application...")
main_app_files = ['app.py', 'main.py', 'application.py', 'run.py']
found_app = None

for app_file in main_app_files:
    if os.path.exists(app_file):
        found_app = app_file
        print(f"✅ Found main app: {app_file}")
        break

if found_app:
    # Try to check if it has Flask app
    with open(found_app, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    if 'Flask(' in content or 'flask.Flask(' in content:
        print("✅ Flask app detected in code")
    else:
        print("❌ No Flask app found in code")
        
    if '@app.route' in content:
        print("✅ Routes detected in code")
    else:
        print("❌ No routes found in code")
else:
    print("❌ No main application file found!")

# 5. Check for common issues
print("\n🔧 Checking for common issues...")
if os.path.exists('templates/base.html'):
    with open('templates/base.html', 'r', encoding='utf-8', errors='ignore') as f:
        base_content = f.read()
    
    if 'bootstrap' in base_content.lower():
        print("✅ Bootstrap detected in base template")
    else:
        print("❌ Bootstrap NOT found in base template")
        
    if '</body>' in base_content:
        print("✅ Base template has proper structure")
    else:
        print("❌ Base template might be incomplete")

print("\n🎯 QUICK DIAGNOSIS COMPLETE!")
print("\n📋 NEXT STEPS:")
print("1. Look for ❌ symbols above")
print("2. Run your app with: python app.py")
print("3. Share any error messages you see")
print("4. If no errors, test the website in browser")