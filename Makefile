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

all: data train

data: $(DATA_FILES)

train:
	pipenv run python -m recommender.train \
		--train-subset train \
		--val-subset val \
		--models-dir ./models/ \
		./data/processed/

setup: $(CREATE_DIRS)

$(DATA_FILES): $(CREATE_DIRS)
	pipenv run python -m recommender.data \
		--output-dir $(OUTPUT_DIR) \
		--subset val 0.05 \
		--subset test 0.05 \
		./data/raw/events_train.csv \
		./data/raw/purchases_train.csv

$(CREATE_DIRS):
	mkdir -p $(CREATE_DIRS)

jupyter: $(DATA_FILES)
	pipenv run jupyter notebook

.PHONY: all data train jupyter setup
