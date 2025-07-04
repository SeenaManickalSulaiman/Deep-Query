FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Prevent Streamlit from showing the local URLs message
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
