import argparse
import csv
import datetime
import functools
import math
import os
import random
import traceback
import itertools
import warnings

from tensorflow.keras import Input, layers, Model
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.metrics import Precision
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
DEFAULT_VIEW_WEIGHT = 0.2
DEFAULT_EMBEDDING_SIZE = 1024
DEFAULT_ENCODER_SIZE = 1024
DEFAULT_BATCH_SIZE = 32


def train(
        data_dir,
        training_subset=None,
        validation_subset=None,
        recommendation_count=DEFAULT_RECOMMENDATION_COUNT,
        view_weight=DEFAULT_VIEW_WEIGHT,
        embedding_size=DEFAULT_EMBEDDING_SIZE,
        encoder_size=DEFAULT_ENCODER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoint_dir=None,
        log_dir=None,
):
    product_count = get_product_count(data_dir=data_dir)
    training_data, training_steps = get_data(
        data_dir=data_dir,
        target_count=recommendation_count,
        product_count=product_count,
        view_weight=view_weight,
        subset=training_subset,
        batch_size=batch_size,
    )
    if validation_subset:
        validation_data, validation_steps = get_data(
            data_dir=data_dir,
            target_count=recommendation_count,
            product_count=product_count,
            view_weight=view_weight,
            subset=validation_subset,
        )
    else:
        validation_data = None
        validation_steps = None
    model, callbacks = get_model(
        recommendation_count=recommendation_count,
        product_count=product_count,
        embedding_size=embedding_size,
        encoder_size=encoder_size,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )
    try:
        model.fit_generator(
            generator=training_data,
            steps_per_epoch=training_steps,
            callbacks=callbacks,
            epochs=1000,
            validation_data=validation_data,
            validation_steps=validation_steps,
        )
    except KeyboardInterrupt:
        print('Training stopped manually.')


def get_product_count(data_dir):
    with open(os.path.join(data_dir, PRODUCT_IDS_FILE), mode='r') as file:
        product_ids = file.read().splitlines()
    return len(product_ids) + 1


def get_data(
        data_dir,
        target_count,
        product_count,
        view_weight,
        subset=None,
        batch_size=1,
):
    series_index = get_series_index(data_dir=data_dir)
    customer_ids = list(get_customer_ids(data_dir=data_dir, subset=subset))
    print(f'Found {len(customer_ids)} IDs in subset "{subset}"')
    line_cache = {}
    batches = get_batches(
        customer_ids=customer_ids,
        series_index=series_index,
        data_dir=data_dir,
        target_count=target_count,
        product_count=product_count,
        view_weight=view_weight,
        batch_size=batch_size,
        line_cache=line_cache,
    )
    step_count = math.ceil(len(customer_ids) / batch_size)
    return batches, step_count


def get_series_index(data_dir):
    with open(os.path.join(data_dir, SERIES_INDEX_FILE), mode='r') as file:
        series_index = {
            row['customer_id']: (row['start_offset'], row['series_length'])
            for row in csv.DictReader(file)
        }
    return series_index


def get_customer_ids(data_dir, subset=None):
    if subset:
        filename = CUSTOMER_IDS_SUBSET_FILE.format(subset=subset)
    else:
        filename = CUSTOMER_IDS_FILE
    with open(os.path.join(data_dir, filename), mode='r') as file:
        for line in file:
            customer_id = line.rstrip('\n')
            yield customer_id


def get_batches(
        customer_ids,
        series_index,
        data_dir,
        target_count,
        product_count,
        view_weight,
        batch_size,
        line_cache,
):
    while True:
        samples = get_samples(
            customer_ids=customer_ids,
            series_index=series_index,
            data_dir=data_dir,
            target_count=target_count,
            product_count=product_count,
            view_weight=view_weight,
            line_cache=line_cache,
        )
        while True:
            batch_samples = list(itertools.islice(samples, batch_size))
            if not batch_samples:
                break
            product_inputs, details_inputs, outputs = zip(*batch_samples)
            batch_inputs = (
                pad_sequences(sequences=product_inputs),
                pad_sequences(sequences=details_inputs),
            )
            yield batch_inputs, numpy.array(outputs)


def get_samples(
        customer_ids,
        series_index,
        data_dir,
        target_count,
        product_count,
        view_weight,
        line_cache,
):
    random.shuffle(customer_ids)
    for customer_id in customer_ids:
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
        product_inputs, details_inputs = get_inputs(
            series=series,
            target_index=target_index,
        )
        outputs = get_outputs(
            series=series,
            target_index=target_index,
            product_count=product_count,
            view_weight=view_weight,
        )
        yield product_inputs, details_inputs, outputs


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
    max_index = get_max_index(series=series, target_count=target_count)
    return random.randrange(1, max_index + 1)


def get_max_index(series, target_count):
    product_set = set()
    max_index = 0
    for reverse_index, step in enumerate(reversed(series)):
        product_set.add(step['product_id'])
        if len(product_set) == target_count:
            max_index = len(series) - reverse_index - 1
    return max_index


def get_inputs(series, target_index):
    if target_index == 0:
        return (
            numpy.zeros(shape=0),
            numpy.zeros(shape=(0, 3)),
        )
    input_series = series[:target_index]
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
        numpy.array(product_inputs),
        numpy.array(details_inputs),
    )


def get_outputs(series, target_index, product_count, view_weight):
    outputs = numpy.zeros(shape=product_count)
    for step in series[target_index:]:
        product_id = int(step['product_id'])
        if int(step['is_purchase']) == 1:
            increment = 1
        else:
            increment = view_weight
        outputs[product_id] += increment
    return outputs / numpy.sum(outputs)


def get_model(
        recommendation_count,
        product_count,
        embedding_size,
        encoder_size,
        log_dir,
        checkpoint_dir,
):
    model = create_model(
        product_count=product_count,
        embedding_size=embedding_size,
        encoder_size=encoder_size,
    )
    top_k_metric = f'precision_top_{recommendation_count}'
    metrics = [
        MaskedPrecision(top_k=1, name='precision'),
        MaskedPrecision(top_k=recommendation_count, name=top_k_metric),
    ]
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=metrics,
    )
    time = datetime.datetime.now().isoformat(sep='-', timespec='seconds')
    callbacks = get_callbacks(
        time=time,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        top_k_metric=top_k_metric,
    )
    save_model_diagram(model=model, time=time, log_dir=log_dir)
    return model, callbacks


def create_model(product_count, embedding_size, encoder_size):
    product_input = Input(shape=(None,), name='products')
    details_input = Input(shape=(None, 3), name='details')
    embedding_layer = layers.Embedding(
        input_dim=product_count,
        output_dim=embedding_size,
        mask_zero=True,
    )
    embedding = embedding_layer(inputs=product_input)
    masked_input = layers.Masking()(inputs=details_input)
    encoder_input = layers.concatenate([embedding, masked_input])
    encoder = layers.Bidirectional(
        layer=layers.LSTM(units=encoder_size),
        name='bidi_lstm',
    )
    encoded_series = encoder(inputs=encoder_input)
    dense = layers.Dense(
        units=encoder_size,
        activation='relu',
        name='dense',
    )
    dense_output = dense(inputs=encoded_series)
    dense_softmax = layers.Dense(
        units=product_count,
        activation='softmax',
        name='dense_softmax',
    )
    model_output = dense_softmax(inputs=dense_output)
    return Model(
        inputs=[product_input, details_input],
        outputs=model_output,
    )


def get_callbacks(time, log_dir=None, checkpoint_dir=None, top_k_metric=None):
    callbacks = [
        SummaryCallback(time=time, log_dir=log_dir),
    ]
    if log_dir:
        logger = CSVLogger(filename=os.path.join(log_dir, f'{time}-log.csv'))
        callbacks.append(logger)
    if checkpoint_dir:
        callbacks += [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f'{time}-latest'),
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f'{time}-best-loss'),
                save_best_only=True,
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f'{time}-best-prec'),
                monitor='precision',
                save_best_only=True,
            ),
        ]
        if top_k_metric:
            top_k_checkpoint = ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f'{time}-best-topk'),
                monitor=top_k_metric,
                save_best_only=True,
            )
            callbacks.append(top_k_checkpoint)
    return callbacks


def save_model_diagram(model, time, log_dir):
    try:
        plot_model(
            to_file=os.path.join(log_dir, f'model-{time}.png'),
            model=model,
        )
        plot_model(
            to_file=os.path.join(log_dir, f'model-{time}-simple.png'),
            model=model,
            show_layer_names=False,
        )
        plot_model(
            to_file=os.path.join(log_dir, f'model-{time}-full.png'),
            model=model,
            show_shapes=True,
        )
        plot_model(
            to_file=os.path.join(log_dir, f'model-{time}-simple-full.png'),
            model=model,
            show_shapes=True,
            show_layer_names=False,
        )
    except ImportError as error:
        traceback.print_tb(tb=error.__traceback__)
        warnings.warn(str(error))


class MaskedPrecision(Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(MaskedPrecision, self).update_state(
            y_true=y_true[:, 1:],
            y_pred=y_pred[:, 1:],
            sample_weight=sample_weight,
        )


class SummaryCallback(Callback):

    def __init__(self, time, log_dir=None):
        super(SummaryCallback, self).__init__()
        self.time = time
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        if self.log_dir:
            summary_path = os.path.join(
                self.log_dir,
                f'{self.time}-summary.txt',
            )
            with open(summary_path, mode='w') as file:
                print_fn = functools.partial(print, file=file)
                self.model.summary(print_fn=print_fn)
        self.model.summary()


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
        '-b', '--batch-size',
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help='training batch size',
        dest='batch_size',
    )
    parser.add_argument(
        '-r', '--recommendations',
        default=DEFAULT_RECOMMENDATION_COUNT,
        type=int,
        help='number of recommendations to make',
        dest='recommendation_count',
    )
    parser.add_argument(
        '-w', '--view-weight',
        default=DEFAULT_VIEW_WEIGHT,
        type=int,
        help='relative importance of a view',
        dest='view_weight',
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
    parser.add_argument(
        '-l', '--log-dir',
        help='training logs directory',
        dest='log_dir',
    )
    parser.add_argument(
        'data_dir',
        help='processed data directory'
    )
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        training_subset=args.training_subset,
        validation_subset=args.validation_subset,
        recommendation_count=args.recommendation_count,
        view_weight=args.view_weight,
        embedding_size=args.embedding_size,
        encoder_size=args.encoder_size,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
