import argparse
import itertools

import numpy
from tensorflow.keras.models import load_model

from recommender.train import (
    DEFAULT_RECOMMENDATION_COUNT,
    DEFAULT_VIEW_WEIGHT,
    get_data,
    get_product_count,
    MaskedPrecision,
)

try:
    from tqdm import tqdm
except ImportError:
    from recommender.utils import tqdm_noop as tqdm


def test(
    data_dir,
    model_path=None,
    training_subset=None,
    testing_subset=None,
    recommendation_count=DEFAULT_RECOMMENDATION_COUNT,
    view_weight=DEFAULT_VIEW_WEIGHT,
):
    product_count = get_product_count(data_dir=data_dir)
    testing_data, testing_steps = get_data(
        data_dir=data_dir,
        target_count=recommendation_count,
        product_count=product_count,
        view_weight=view_weight,
        subset=testing_subset,
    )
    if model_path is not None:
        model = load_model(filepath=model_path)
        model.evaluate_generator(
            generator=testing_data,
            steps_per_epoch=testing_steps,
        )
    training_data, training_steps = get_data(
        data_dir=data_dir,
        target_count=recommendation_count,
        product_count=product_count,
        view_weight=view_weight,
        subset=training_subset,
        batch_size=1,
    )
    baseline = train_baseline(
        data=training_data,
        steps=training_steps,
        product_count=product_count,
    )
    test_baseline(
        baseline=baseline,
        data=testing_data,
        steps=testing_steps,
        recommendation_count=recommendation_count,
    )


def train_baseline(data, steps, product_count):
    baseline = numpy.zeros(shape=product_count)
    batches = tqdm(
        itertools.islice(data, steps),
        desc='Training baseline model',
        total=steps,
        dynamic_ncols=True,
    )
    for batch in batches:
        inputs, outputs = batch
        baseline += numpy.sum(outputs, axis=0)
    return baseline / numpy.sum(baseline)


def test_baseline(baseline, data, steps, recommendation_count):
    top_k_metric = f'precision_top_{recommendation_count}'
    metrics = [
        MaskedPrecision(top_k=1, name='precision'),
        MaskedPrecision(top_k=recommendation_count, name=top_k_metric),
    ]
    for batch in itertools.islice(data, steps):
        inputs, outputs = batch
        for metric in metrics:
            y_pred = numpy.repeat(baseline[None], repeats=len(outputs), axis=0)
            metric.update_state(y_true=outputs, y_pred=y_pred)
    for metric in metrics:
        print(f'Baseline {metric.name}: {metric.result().numpy()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '-s', '--test-subset',
        help='testing subset identifier',
        dest='testing_subset',
    )
    parser.add_argument(
        '-m', '--model-path',
        help='model checkpoint path',
        dest='model_path',
    )
    parser.add_argument(
        'data_dir',
        help='processed data directory'
    )
    args = parser.parse_args()
    test(
        data_dir=args.data_dir,
        model_path=args.model_path,
        training_subset=args.training_subset,
        testing_subset=args.testing_subset,
        recommendation_count=args.recommendation_count,
        view_weight=args.view_weight,
    )
