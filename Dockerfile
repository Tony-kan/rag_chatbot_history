
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# FROM pytorch/pytoch:1.12.1-cuda11.3-cudnn8-runtime-python3.9
FROM python:3.11

# Install PyTorch
# RUN pip install torch torchvision

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--port", "8000"]