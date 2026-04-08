import subprocess
import time
import sys
import os

def run_project():
    print("🚀 Starting PDF Chatbot Project...")
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("⚠️ Warning: .env file not found. AI features might not work.")

    try:
        # Start Backend (FastAPI)
        print("📡 Starting Backend at http://localhost:8000...")
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            # Removed PIPE to allow output to flow to console
        )

        # Wait a bit for backend to initialize
        time.sleep(3)

        # Start Frontend (Streamlit)
        print("🎨 Starting Frontend at http://localhost:8501...")
        frontend_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
            # Removed PIPE to allow output to flow to console
        )

        print("\n✅ Project is running!")
        print("- Frontend: http://localhost:8501")
        print("- Backend API: http://localhost:8000")
        print("- API Documentation: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both services.\n")

        # Monitor both processes
        while True:
            if backend_proc.poll() is not None:
                print("❌ Backend process stopped.")
                break
            if frontend_proc.poll() is not None:
                print("❌ Frontend process stopped.")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("👋 Goodbye!")

if __name__ == "__main__":
    run_project()
