FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import RobertaModel; RobertaModel.from_pretrained('roberta-base')"
COPY . /app/
CMD ["python", "src/ERFD.py", "--dataset_name", "politifact", "--epochs", "5"]
