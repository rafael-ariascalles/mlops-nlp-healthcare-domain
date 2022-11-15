FROM python:3.9
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt
EXPOSE 8000
ENV ICD_SERVICE='ICD:8000'
ENV NER_SERVICE='NER:8000'
COPY . .
CMD ["uvicorn", "service:app","--host", "0.0.0.0","--port", "8000"]