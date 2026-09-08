#!/usr/bin/env python3
import os
import sys
import subprocess

print("=" * 60)
print("🔍 WALL CRACK DETECTION - PROJECT VERIFICATION")
print("=" * 60)

checks_passed = 0
checks_total = 0

print("\n📋 Checking required files...")
required_files = [
    'app.py',
    'predict.py',
    'cnn_wall_crack_model.h5',
    'best_wall_crack_model.h5',
    'dummy.jpg'
]

for file in required_files:
    checks_total += 1
    if os.path.exists(file):
        print(f"  ✅ {file}")
        checks_passed += 1
    else:
        print(f"  ❌ {file} - MISSING!")

print("\n📦 Checking Python dependencies...")
dependencies = [
    ('tensorflow', 'TensorFlow'),
    ('keras', 'Keras'),
    ('streamlit', 'Streamlit'),
    ('cv2', 'OpenCV'),
    ('PIL', 'Pillow'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'Scikit-learn')
]

for module, name in dependencies:
    checks_total += 1
    try:
        __import__(module)
        print(f"  ✅ {name}")
        checks_passed += 1
    except ImportError:
        print(f"  ❌ {name} - NOT INSTALLED")

print("\n" + "=" * 60)
print(f"📊 RESULTS: {checks_passed}/{checks_total} checks passed")
print("=" * 60)

if checks_passed >= (checks_total * 0.8):
    print("\n✅ Project is ready! Run: streamlit run app.py")
else:
    print("\n⚠️  Fix missing items above, then try again.")
