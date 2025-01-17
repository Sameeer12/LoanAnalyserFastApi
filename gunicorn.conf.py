import os

command = "gunicorn"
pythonpath = "src"
bind = f"0.0.0.0:{os.getenv('PORT', 8000)}"
workers = int(os.getenv("GUNICORN_WORKERS", 2))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300