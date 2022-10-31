FROM python:3.9
RUN  python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN  pip install -r requirements.txt
RUN  mkdir models
COPY get_models.py .
RUN  python get_models.py
COPY copy_models_s3.sh .
RUN bash copy_models_s3.sh
COPY . .
CMD ["uvicorn", "api:app","--host", "0.0.0.0"]