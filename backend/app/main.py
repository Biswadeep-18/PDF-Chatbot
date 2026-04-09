from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import endpoints, auth_routes
from .core.config import settings
from .services.mongodb_service import db_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Backend for PDF Chatbot"
)

@app.on_event("startup")
async def startup_event():
    try:
        db_service.connect()
        print("Backend startup successful")
    except Exception as e:
        print(f"Backend startup failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)
app.include_router(endpoints.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
