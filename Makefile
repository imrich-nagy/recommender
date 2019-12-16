CREATE_DIRS = ./data/raw/ ./data/processed/ ./models/ ./logs/
DATA_FILES = \
	./data/processed/series.csv \
	./data/processed/series.index.csv \
	./data/processed/customer_ids.txt \
	./data/processed/product_ids.txt \
	./data/processed/customer_ids.train.txt \
	./data/processed/customer_ids.test.txt \
	./data/processed/customer_ids.val.txt
OUTPUT_DIR = ./data/processed/

TRAIN_ARGS = \
	--train-subset train \
	--val-subset val \
	--models-dir ./models/ \
	--log-dir ./logs/ \
	./data/processed/


all: data train

data: $(CREATE_DIRS)
	pipenv run python -m recommender.data \
		--output-dir $(OUTPUT_DIR) \
		--subset val 0.05 \
		--subset test 0.05 \
		./data/raw/events_train.csv \
		./data/raw/purchases_train.csv

train:
	pipenv run python -m recommender.train $(TRAIN_ARGS)

train-gpu:
	docker run --rm --interactive --tty \
		--gpus all \
		--workdir /home \
		--volume "$(shell pwd)":/home \
		--user "$(shell id -u)":"$(shell id -g)" \
		tensorflow/tensorflow:2.0.0-gpu-py3 \
		python -m recommender.train $(TRAIN_ARGS)

test:
	pipenv run python -m recommender.test \
		--train-subset train \
		--test-subset test \
		--model-path $(MODEL) \
		./data/processed

setup: $(CREATE_DIRS)

$(DATA_FILES): data

$(CREATE_DIRS):
	mkdir -p $(CREATE_DIRS)

jupyter:
	pipenv run jupyter notebook

.PHONY: all data train jupyter setup
