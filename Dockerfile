FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit pillow numpy matplotlib

COPY . .
# Hugging Face Spaces runs containers as a non-root user; torchvision caches weights under $HOME.
ENV HOME=/tmp
EXPOSE 8501
# Spaces embeds the app in an iframe where Streamlit's XSRF cookie is blocked, which breaks uploads.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
