"""
Quick deployment script for Android/React Native.
Automates model conversion and package creation.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import subprocess
import sys
from pathlib import Path
import shutil

def main():
    print("=" * 70)
    print("📱 Sinhala Sign Language - Android Deployment")
    print("=" * 70)
    print()
    
    # Check if running from correct directory
    current_dir = Path.cwd()
    if current_dir.name != 'src':
        print("⚠️  Please run this script from the 'src' directory")
        print(f"   Current directory: {current_dir}")
        sys.exit(1)
    
    # Check if model exists
    model_path = Path('../models/checkpoint_best.pth')
    if not model_path.exists():
        print("❌ Model not found!")
        print(f"   Looking for: {model_path}")
        print("\n   Please train a model first:")
        print("   python quick_train.py")
        sys.exit(1)
    
    print("✓ Model found:", model_path)
    print()
    
    # Ask user what to deploy
    print("Select deployment option:")
    print("  1. Android (TensorFlow Lite) - Recommended")
    print("  2. React Native Bridge API (Local Server)")
    print("  3. Both")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    deploy_android = choice in ["1", "3"]
    deploy_api = choice in ["2", "3"]
    
    print()
    print("=" * 70)
    print("Starting deployment...")
    print("=" * 70)
    print()
    
    # Deploy for Android
    if deploy_android:
        print("📱 Creating Android deployment package...")
        print()
        
        try:
            cmd = [
                sys.executable,
                "convert_to_mobile.py",
                "--model_path", str(model_path),
                "--android_package",
                "--output_dir", "../models/android_deployment"
            ]
            
            subprocess.run(cmd, check=True)
            
            print()
            print("✓ Android package created!")
            print(f"  Location: {Path('../models/android_deployment').absolute()}")
            print()
            
            # Show next steps
            print("=" * 70)
            print("📋 Next Steps for Android:")
            print("=" * 70)
            print()
            print("1. Copy model files to React Native project:")
            print("   cp ../models/android_deployment/model.tflite <your-rn-project>/android/app/src/main/assets/")
            print("   cp ../models/android_deployment/labels.txt <your-rn-project>/android/app/src/main/assets/")
            print()
            print("2. Follow integration guide:")
            print("   ../models/android_deployment/ANDROID_INTEGRATION.md")
            print()
            print("3. Or use our React Native template:")
            print("   See: REACT_NATIVE_GUIDE.md")
            print()
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Android deployment failed: {e}")
            sys.exit(1)
    
    # Deploy API server
    if deploy_api:
        print("🌐 Setting up React Native Bridge API...")
        print()
        
        print("To start the API server:")
        print(f"  python react_native_bridge.py --model_path {model_path}")
        print()
        print("API Endpoints:")
        print("  GET  /health         - Health check")
        print("  GET  /labels         - Get all labels")
        print("  POST /predict_frame  - Predict from single frame")
        print("  POST /predict_video  - Predict from video")
        print("  POST /reset_buffer   - Reset frame buffer")
        print()
        print("React Native can connect to: http://<your-ip>:5000")
        print()
        
        start_now = input("Start API server now? (y/n) [default: n]: ").strip().lower()
        
        if start_now == 'y':
            try:
                cmd = [
                    sys.executable,
                    "react_native_bridge.py",
                    "--model_path", str(model_path),
                    "--host", "0.0.0.0",
                    "--port", "5000"
                ]
                
                print("\nStarting server...")
                print("Press Ctrl+C to stop")
                print()
                
                subprocess.run(cmd)
                
            except KeyboardInterrupt:
                print("\n\nServer stopped.")
            except subprocess.CalledProcessError as e:
                print(f"❌ API server failed: {e}")
    
    print()
    print("=" * 70)
    print("✓ Deployment Complete!")
    print("=" * 70)
    print()
    print("📚 Documentation:")
    print("  - ON_DEVICE_DEPLOYMENT.md - On-device processing guide")
    print("  - REACT_NATIVE_GUIDE.md - React Native integration")
    print("  - TRAINING_GUIDE.md - Model training guide")
    print()
    print("🚀 Your app is ready for Android deployment with:")
    print("  ✅ Real-time sign language recognition")
    print("  ✅ On-device processing (no internet)")
    print("  ✅ Privacy-preserving")
    print("  ✅ React Native frontend ready")
    print()


if __name__ == "__main__":
    main()
