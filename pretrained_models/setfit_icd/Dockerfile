FROM python:3.9
RUN  python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN  pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app","--host", "0.0.0.0"]