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

data: $(DATA_FILES)

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

.PHONY: data jupyter setup
