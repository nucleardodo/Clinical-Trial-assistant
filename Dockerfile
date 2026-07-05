FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     HOME=/tmp     XDG_CONFIG_HOME=/tmp     STREAMLIT_HOME=/tmp     STREAMLIT_CONFIG_DIR=/tmp/.streamlit     STREAMLIT_BROWSER_GATHER_USAGE_STATS=false     STREAMLIT_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /tmp/.streamlit && printf '[server]
headless = true
port = 7860
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
' > /tmp/.streamlit/config.toml

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
