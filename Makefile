data: setup
	pipenv run python -m recommender.data --output-dir ./data/processed/ \
		./data/raw/events_train.csv ./data/raw/purchases_train.csv

jupyter: setup
	pipenv run jupyter notebook

setup:
	mkdir -p ./data/raw/ ./data/processed/

.PHONY: data jupyter setup
