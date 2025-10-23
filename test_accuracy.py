import pickle
import os
import numpy as np  # ADD THIS LINE

def check_model_status():
    print("🔍 CHECKING MODEL STATUS...")
    
    # Check if model file exists
    if not os.path.exists("encodings/encodings.pickle"):
        print("❌ MODEL FILE NOT FOUND")
        return
    
    # Load and analyze model
    with open("encodings/encodings.pickle", "rb") as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Total students in model: {len(model.get('encodings', []))}")
    print(f"👥 Students: {model.get('names', [])}")
    print(f"🕒 Last trained: {model.get('last_trained', 'Unknown')}")
    
    # Check encoding quality
    encodings = model.get('encodings', [])
    if encodings:
        print(f"📏 Encoding dimensions: {encodings[0].shape}")
        print(f"💪 Encoding strength: {np.linalg.norm(encodings[0]):.2f}")

if __name__ == "__main__":
    check_model_status()