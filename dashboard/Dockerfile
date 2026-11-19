FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "streamlit>=1.38,<2" "requests>=2.32,<3" "pandas>=2.2,<3"

COPY dashboard/ dashboard/

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
