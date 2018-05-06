FROM pytorch/pytorch:latest

RUN adduser --disabled-password --gecos '' --shell /bin/bash vinh
USER vinh
WORKDIR /home/vinh/code
ADD . .

ENTRYPOINT ["python", "src/MNIST/cnn.py"]
