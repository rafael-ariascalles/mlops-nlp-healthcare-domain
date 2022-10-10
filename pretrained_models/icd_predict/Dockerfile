FROM python:3.6
COPY requirements.txt .
RUN  python3 -m pip install --upgrade pip
RUN  pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app","--host", "0.0.0.0"]

#docker build -t 'icd_prediction_model' .
#docker run -p 8000:8000 icd_prediction_model