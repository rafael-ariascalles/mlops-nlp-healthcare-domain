FROM python:3.10
RUN pip install torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt .
RUN  pip install -r requirements.txt
EXPOSE 8081
ENV NVIDIA_TRITON_ENDPOINT='triton-endpoint:8000'
COPY . .
CMD ["uvicorn", "main:app","--host", "0.0.0.0","--port", "8081"]