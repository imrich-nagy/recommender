import argparse
import itertools

import numpy

from recommender.train import (
    DEFAULT_RECOMMENDATION_COUNT,
    DEFAULT_VIEW_WEIGHT,
    get_data,
    get_product_count,
    MaskedPrecision,
)


def test(
    data_dir,
    training_subset=None,
    testing_subset=None,
    recommendation_count=DEFAULT_RECOMMENDATION_COUNT,
    view_weight=DEFAULT_VIEW_WEIGHT,
):
    product_count = get_product_count(data_dir=data_dir)
    training_data, training_steps = get_data(
        data_dir=data_dir,
        target_count=recommendation_count,
        product_count=product_count,
        view_weight=view_weight,
        subset=training_subset,
        do_filter=False,
    )
    baseline = train_baseline(
        data=training_data,
        steps=training_steps,
        product_count=product_count,
    )
    testing_data, testing_steps = get_data(
        data_dir=data_dir,
        target_count=recommendation_count,
        product_count=product_count,
        view_weight=view_weight,
        subset=testing_subset,
    )
    test_baseline(
        baseline=baseline,
        data=testing_data,
        steps=testing_steps,
        recommendation_count=recommendation_count,
    )


def train_baseline(data, steps, product_count):
    baseline = numpy.zeros(shape=product_count)
    for batch in itertools.islice(data, steps):
        inputs, outputs = batch
        for output in outputs:
            baseline += output
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
        'data_dir',
        help='processed data directory'
    )
    args = parser.parse_args()
    test(
        data_dir=args.data_dir,
        training_subset=args.training_subset,
        testing_subset=args.testing_subset,
        recommendation_count=args.recommendation_count,
        view_weight=args.view_weight,
    )
