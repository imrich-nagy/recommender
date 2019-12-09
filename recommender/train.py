import argparse
import csv
import os
import random
import traceback
import itertools
import warnings

from tensorflow.keras import Input, layers, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.metrics import Precision
from tensorflow.keras.utils import plot_model
import numpy

from recommender.data import (
    CUSTOMER_IDS_FILE,
    CUSTOMER_IDS_SUBSET_FILE,
    PRODUCT_IDS_FILE,
    SERIES_CSV_FILE,
    SERIES_FIELDS,
    SERIES_INDEX_FILE,
)


DEFAULT_RECOMMENDATION_COUNT = 10
DEFAULT_EMBEDDING_SIZE = 256
DEFAULT_ENCODER_SIZE = 256


def train(
        data_dir,
        training_subset=None,
        validation_subset=None,
        recommendation_count=DEFAULT_RECOMMENDATION_COUNT,
        embedding_size=DEFAULT_EMBEDDING_SIZE,
        encoder_size=DEFAULT_ENCODER_SIZE,
        checkpoint_dir=None,
):
    product_count = get_product_count(data_dir=data_dir)
    training_data, training_steps = get_data(
        data_dir=data_dir,
        target_count=recommendation_count,
        product_count=product_count,
        subset=training_subset,
    )
    if validation_subset:
        validation_data, validation_steps = get_data(
            data_dir=data_dir,
            target_count=recommendation_count,
            product_count=product_count,
            subset=validation_subset,
        )
    else:
        validation_data = None
        validation_steps = None
    model = create_model(
        product_count=product_count,
        embedding_size=embedding_size,
        encoder_size=encoder_size,
    )
    save_model_diagram(model=model)
    top_k_metric = f'precision_top_{recommendation_count}'
    metrics = [
        Precision(top_k=1, name='precision'),
        Precision(top_k=recommendation_count, name=top_k_metric),
    ]
    model.compile(
        optimizer='adam',
        loss='cosine_similarity',
        metrics=metrics,
    )
    callbacks = [
        DiagramCallback(),
    ]
    if checkpoint_dir:
        callbacks += [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'last-epoch.model'),
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'last-100s.model'),
                save_freq=100,
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best-loss.model'),
                save_best_only=True,
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best-prec.model'),
                monitor='precision',
                save_best_only=True,
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best-topk.model'),
                monitor=top_k_metric,
                save_best_only=True,
            ),
        ]
    model.fit_generator(
        generator=training_data,
        steps_per_epoch=training_steps,
        callbacks=callbacks,
        epochs=100,
        validation_data=validation_data,
        validation_steps=validation_steps,
    )
    model.summary()


def get_product_count(data_dir):
    with open(os.path.join(data_dir, PRODUCT_IDS_FILE), mode='r') as file:
        product_ids = file.read().splitlines()
    return len(product_ids) + 1


def get_data(data_dir, target_count, product_count, subset=None):
    if subset:
        filename = CUSTOMER_IDS_SUBSET_FILE.format(subset=subset)
    else:
        filename = CUSTOMER_IDS_FILE
    with open(os.path.join(data_dir, filename), mode='r') as file:
        customer_ids = file.read().splitlines()
    with open(os.path.join(data_dir, SERIES_INDEX_FILE), mode='r') as file:
        series_index = {
            row['customer_id']: (row['start_offset'], row['series_length'])
            for row in csv.DictReader(file)
        }
    samples = get_batches(
        customer_ids=customer_ids,
        series_index=series_index,
        data_dir=data_dir,
        target_count=target_count,
        product_count=product_count,
    )
    return samples, len(customer_ids)


def get_batches(
        customer_ids,
        series_index,
        data_dir,
        target_count,
        product_count,
):
    line_cache = {}
    while True:
        random.shuffle(customer_ids)
        for customer_id in customer_ids:
            sample = get_sample(
                customer_id=customer_id,
                series_index=series_index,
                data_dir=data_dir,
                target_count=target_count,
                product_count=product_count,
                line_cache=line_cache,
            )
            yield sample


def get_sample(
        customer_id,
        series_index,
        data_dir,
        target_count,
        product_count,
        line_cache,
):
    series = get_series(
        customer_id=customer_id,
        series_index=series_index,
        data_dir=data_dir,
        line_cache=line_cache,
    )
    target_index = get_target_index(
        series=series,
        target_count=target_count,
    )
    inputs = get_inputs(
        series=series,
        target_index=target_index,
    )
    outputs = get_outputs(
        series=series,
        target_index=target_index,
        product_count=product_count,
    )
    return inputs, outputs


def get_series(customer_id, series_index, data_dir, line_cache):
    with open(os.path.join(data_dir, SERIES_CSV_FILE), mode='r') as file:
        start_offset, series_length = series_index[customer_id]
        start_offset = int(start_offset)
        series_length = int(series_length)
        seek_line(file=file, line=start_offset + 1, cache=line_cache)
        reader = csv.DictReader(file, fieldnames=SERIES_FIELDS)
        return list(itertools.islice(reader, series_length))


def seek_line(file, line, cache):
    file.seek(0)
    if line in cache:
        file.seek(cache[line])
        return
    for _ in range(line):
        file.readline()
    cache[line] = file.tell()


def get_target_index(series, target_count):
    purchase_count = 0
    target_index = 0
    for reverse_index, step in enumerate(reversed(series)):
        if int(step['is_purchase']) == 1:
            purchase_count += 1
        if purchase_count == target_count:
            target_index = len(series) - reverse_index - 1
    return target_index


def get_inputs(series, target_index):
    input_series = series[:target_index]
    if not input_series:
        details_inputs = [(0, 0, -1)]
        return (
            numpy.zeros(shape=(1, 1)),
            numpy.array(details_inputs)[None],
        )
    product_inputs = []
    details_inputs = []
    time_offset = float(input_series[-1]['timestamp'])
    for step in input_series:
        product_inputs.append(int(step['product_id']))
        relative_time = float(step['timestamp']) - time_offset
        price = float(step['price'])
        is_purchase = int(step['is_purchase'])
        details_inputs.append((relative_time, price, is_purchase))
    return (
        numpy.array(product_inputs)[None],
        numpy.array(details_inputs)[None],
    )


def get_outputs(series, target_index, product_count):
    targets = [
        int(step['product_id']) for step in series[target_index:]
        if int(step['is_purchase']) == 1
    ]
    outputs = numpy.zeros(shape=product_count)
    outputs[targets] = 1
    return outputs[None]


def create_model(product_count, embedding_size, encoder_size):
    product_input = Input(shape=(None,), name='products')
    details_input = Input(shape=(None, 3), name='details')
    embedding_layer = layers.Embedding(
        input_dim=product_count,
        output_dim=embedding_size,
    )
    embedding = embedding_layer(inputs=product_input)
    encoder_input = layers.concatenate([embedding, details_input])
    encoder = layers.LSTM(units=encoder_size, name='encoder')
    encoded_series = encoder(inputs=encoder_input)
    dense = layers.Dense(units=product_count, activation='softmax')
    model_output = dense(inputs=encoded_series)
    return Model(
        inputs=[product_input, details_input],
        outputs=model_output,
    )


def save_model_diagram(model, name=None):
    if name:
        filename = f'model-{name}.png'
    else:
        filename = 'model.png'
    try:
        plot_model(to_file=filename, model=model, show_shapes=True)
    except ImportError as error:
        traceback.print_tb(tb=error.__traceback__)
        warnings.warn(str(error))


class DiagramCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        if batch == 0:
            save_model_diagram(model=self.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--embedding',
        default=DEFAULT_EMBEDDING_SIZE,
        type=int,
        help='product embedding size',
        dest='embedding_size',
    )
    parser.add_argument(
        '-c', '--encoder',
        default=DEFAULT_ENCODER_SIZE,
        type=int,
        help='encoder vector size',
        dest='encoder_size',
    )
    parser.add_argument(
        '-r', '--recommendations',
        default=DEFAULT_RECOMMENDATION_COUNT,
        type=int,
        help='number of recommendations to make',
        dest='recommendation_count',
    )
    parser.add_argument(
        '-t', '--train-subset',
        help='training subset identifier',
        dest='training_subset',
    )
    parser.add_argument(
        '-v', '--val-subset',
        help='validation subset identifier',
        dest='validation_subset',
    )
    parser.add_argument(
        '-m', '--models-dir',
        help='model checkpoint directory',
        dest='checkpoint_dir',
    )
    parser.add_argument('data_dir')
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        training_subset=args.training_subset,
        validation_subset=args.validation_subset,
        recommendation_count=args.recommendation_count,
        embedding_size=args.embedding_size,
        encoder_size=args.encoder_size,
        checkpoint_dir=args.checkpoint_dir,
    )
