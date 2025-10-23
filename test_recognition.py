import face_recognition
import pickle
import numpy as np

def debug_recognition():
    print("🐛 DEBUGGING RECOGNITION LOGIC...")
    
    # Load model
    with open("encodings/encodings.pickle", "rb") as f:
        model = pickle.load(f)
    
    # Test the two same-person entries
    nithya_idx = model['names'].index('nithya (cs33)')
    eve_idx = model['names'].index('eve (cs12)')
    
    nithya_encoding = model['encodings'][nithya_idx]
    eve_encoding = model['encodings'][eve_idx]
    
    distance = face_recognition.face_distance([nithya_encoding], eve_encoding)[0]
    confidence = (1 - distance) * 100
    
    print(f"📊 Nithya vs Eve (SAME PERSON):")
    print(f"   Distance: {distance:.4f}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"   Default threshold: 0.6")
    print(f"   Should recognize as same: {distance <= 0.6}")
    
    # Check what threshold your app is using
    print(f"\n🔧 Current recognition settings:")
    print(f"   If distance <= 0.6 → Same person")
    print(f"   Your distance: {distance:.4f} → {'SAME' if distance <= 0.6 else 'DIFFERENT'}")

if __name__ == "__main__":
    debug_recognition()