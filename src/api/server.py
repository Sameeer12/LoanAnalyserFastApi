import uvicorn
from .routes import app, config

def start_server():
    """Start the API server"""
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )