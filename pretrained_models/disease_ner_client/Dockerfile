FROM pytorch/pytorch
COPY requirements.txt .
RUN  pip install -r requirements.txt
EXPOSE 8081
ENV NVIDIA_TRITON_ENDPOINT='triton-endpoint:8000'
COPY . .
CMD ["uvicorn", "main:app","--host", "0.0.0.0","--port", "8081"]