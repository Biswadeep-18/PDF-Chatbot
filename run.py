import subprocess
import time
import sys
import os

def run_project():
    print("Starting PDF Chatbot Project...")
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("Warning: .env file not found. AI features might not work.")

    try:
        # Start Backend (FastAPI)
        print("Starting Backend at http://127.0.0.1:8001...")
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8001"],
        )

        # Wait a bit for backend to initialize
        time.sleep(4)

        # Quick check if backend is alive
        if backend_proc.poll() is not None:
            print("Error: Backend failed to start. Check if port 8001 is already in use.")
            return

        # Start Frontend (Streamlit)
        print("Starting Frontend at http://localhost:8501...")
        frontend_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py"],
        )

        print("\nProject is running!")
        print("- Frontend: http://localhost:8501")
        print("- Backend API: http://127.0.0.1:8001")
        print("- API Documentation: http://127.0.0.1:8001/docs")
        print("\nPress Ctrl+C to stop both services.\n")

        # Monitor both processes
        while True:
            if backend_proc.poll() is not None:
                print("Backend process stopped.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend process stopped.")
                break
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    run_project()
