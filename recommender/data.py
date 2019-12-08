import argparse
import collections
import csv
import datetime
import os

try:
    from tqdm import tqdm
except ImportError:
    from recommender.utils import tqdm_noop as tqdm


EVENTS_COUNT = 14614386
PURCHASES_COUNT = 188713

CUSTOMER_IDS_FILE = 'customer_ids.txt'
PRODUCT_IDS_FILE = 'product_ids.txt'
SERIES_CSV_FILE = 'series.csv'
SERIES_INDEX_FILE = 'series.index.csv'

DEFAULT_MIN_EVENTS = 10
DEFAULT_MIN_PURCHASES = 10
DEFAULT_MIN_PRODUCTS = 10

SERIES_FIELDS = [
    'timestamp',
    'product_id',
    'is_purchase',
    'price',
]
SERIES_INDEX_FIELDS = [
    'customer_id',
    'start_offset',
    'series_length',
]


def process_data(
        events_data,
        purchases_data,
        output_dir,
        min_events=DEFAULT_MIN_EVENTS,
        min_purchases=DEFAULT_MIN_PURCHASES,
        min_products=DEFAULT_MIN_PRODUCTS,
):
    """
    Preprocess data for training.
    """
    try:
        customer_ids = get_customers(
            events_data=events_data,
            purchases_data=purchases_data,
            min_events=min_events,
            min_purchases=min_purchases,
        )
        write_ids(
            id_list=customer_ids,
            output_dir=output_dir,
            filename=CUSTOMER_IDS_FILE,
        )
        product_ids = get_products(
            events_data=events_data,
            purchases_data=purchases_data,
            customer_ids=customer_ids,
            min_products=min_products,
        )
        write_ids(
            id_list=product_ids,
            output_dir=output_dir,
            filename=PRODUCT_IDS_FILE,
        )
        encode_series(
            events_data=events_data,
            purchases_data=purchases_data,
            customer_ids=customer_ids,
            product_ids=product_ids,
            output_dir=output_dir,
        )
    finally:
        events_data.close()
        purchases_data.close()


def get_customers(events_data, purchases_data, min_events, min_purchases):
    """
    Get IDs of customers that meet the minimum required events and purchases.
    """
    seek_beginning(events_data, purchases_data)
    events_counter = count_customers(
        file=events_data,
        total=EVENTS_COUNT,
        desc='Retrieving customer IDs from events',
    )
    print(f'{len(events_counter)} customer IDs found in events.')
    purchases_counter = count_customers(
        file=purchases_data,
        total=PURCHASES_COUNT,
        desc='Retrieving customer IDs from purchases',
    )
    print(f'{len(purchases_counter)} customer IDs found in purchases.')
    total_counter = events_counter + purchases_counter
    events_ids = filter_counts(
        counter=total_counter,
        min_count=min_events,
    )
    print(
        f'{len(events_ids)} customers are above '
        f'{min_events} events minimum'
    )
    purchases_ids = filter_counts(
        counter=purchases_counter,
        min_count=min_purchases,
    )
    print(
        f'{len(purchases_ids)} customers are above '
        f'{min_purchases} purchases minimum'
    )
    customer_ids = list(events_ids & purchases_ids)
    print(f'Retrieved {len(customer_ids)} customer IDs in total.')
    return customer_ids


def count_customers(file, total=None, desc=None):
    """
    Count occurrences of distinct customer IDs.
    """
    reader = tqdm(
        csv.DictReader(file),
        desc=desc,
        total=total,
        dynamic_ncols=True,
    )
    customer_ids = (row['customer_id'] for row in reader)
    counter = collections.Counter(customer_ids)
    return counter


def get_products(events_data, purchases_data, customer_ids, min_products):
    """
    Get IDs of products that meet the minimum required number of occurrences
    for specified customer.
    """
    seek_beginning(events_data, purchases_data)
    events_counter = count_products(
        file=events_data,
        customer_ids=customer_ids,
        total=EVENTS_COUNT,
        desc='Retrieving product IDs from events',
    )
    print(f'{len(events_counter)} product IDs found in events.')
    purchases_counter = count_products(
        file=purchases_data,
        customer_ids=customer_ids,
        total=PURCHASES_COUNT,
        desc='Retrieving product IDs from purchases',
    )
    print(f'{len(purchases_counter)} product IDs found in purchases.')
    total_counter = events_counter + purchases_counter
    product_ids = filter_counts(
        counter=total_counter,
        min_count=min_products,
    )
    print(
        f'{len(product_ids)} products are above '
        f'{min_products} occurrences minimum'
    )
    print(f'Found {len(product_ids)} product IDs in total.')
    return product_ids


def count_products(file, customer_ids, total=None, desc=None):
    """
    Count occurrences of distinct product IDs for specified customers.
    """
    reader = tqdm(
        csv.DictReader(file),
        total=total,
        desc=desc,
        dynamic_ncols=True,
    )
    customer_set = set(customer_ids)
    product_ids = (
        row['product_id'] for row in reader
        if row['customer_id'] in customer_set
    )
    counter = collections.Counter(product_ids)
    return counter


def filter_counts(counter, min_count):
    """
    Filter counter items by minimum count.
    """
    return set(key for key, value in counter.items() if value >= min_count)


def write_ids(id_list, output_dir, filename):
    """
    Write IDs to a text file.
    """
    with open(os.path.join(output_dir, filename), mode='w') as file:
        for line in id_list:
            file.write(f'{line}\n')


def encode_series(
        events_data,
        purchases_data,
        customer_ids,
        product_ids,
        output_dir,
):
    """
    Encode data into separate time series for each customer.
    """
    seek_beginning(events_data, purchases_data)
    series_dict = {customer_id: [] for customer_id in customer_ids}
    series_dict = encode_file(
        file=events_data,
        product_ids=product_ids,
        series_dict=series_dict,
        total=EVENTS_COUNT,
        desc='Encoding time series from event data',
    )
    series_dict = encode_file(
        file=purchases_data,
        product_ids=product_ids,
        series_dict=series_dict,
        total=PURCHASES_COUNT,
        desc='Encoding time series from purchase data',
    )
    index_list = []
    position = 0
    with open(os.path.join(output_dir, SERIES_CSV_FILE), mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=SERIES_FIELDS)
        writer.writeheader()
        customer_ids_progress = tqdm(
            customer_ids,
            desc='Writing time series data',
        )
        for customer_id in customer_ids_progress:
            series = series_dict[customer_id]
            series.sort(key=lambda step: step['timestamp'])
            writer.writerows(series)
            index_list.append((customer_id, position, len(series)))
            position += len(series)
    with open(os.path.join(output_dir, SERIES_INDEX_FILE), mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(SERIES_INDEX_FIELDS)
        index_list_progress = tqdm(
            index_list,
            desc='Writing time series index',
        )
        for index_data in index_list_progress:
            writer.writerow(index_data)


def encode_file(
        file,
        product_ids,
        series_dict,
        total=None,
        desc=None,
):
    """
    Get time series for each customer from specified file.
    """
    reader = tqdm(
        csv.DictReader(file),
        total=total,
        desc=desc,
        dynamic_ncols=True,
    )
    product_dict = {
        product_id: index + 1 for index, product_id in enumerate(product_ids)
    }
    for row in reader:
        customer_id = row['customer_id']
        if customer_id in series_dict:
            step_dict = encode_step(row=row, product_dict=product_dict)
            series_dict[customer_id].append(step_dict)
    return series_dict


def encode_step(row, product_dict):
    """
    Encode a single step of the time series.
    """
    timestamp = encode_timestamp(timestamp=row['timestamp'])
    product_id = product_dict.get(row['product_id'], 0)
    is_purchase = 1 if row['event_type'] == 'purchase_item' else 0
    try:
        price = float(row['price'])
    except ValueError:
        price = 0.0
    return {
        'timestamp': timestamp,
        'product_id': product_id,
        'is_purchase': is_purchase,
        'price': price,
    }


def encode_timestamp(timestamp):
    """
    Encode date string into Unix timestamp.
    """
    date_string = timestamp.replace(' UTC', '')
    date_object = datetime.datetime.strptime(
        date_string,
        '%Y-%m-%d %H:%M:%S.%f',
    )
    return date_object.timestamp()


def seek_beginning(*files):
    """
    Seek to the beginning of all specified files.
    """
    for file in files:
        file.seek(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-e', '--events',
        default=DEFAULT_MIN_EVENTS,
        type=int,
        help='minimum number of total events or purchases',
        dest='min_events',
    )
    parser.add_argument(
        '-p', '--purchases',
        default=DEFAULT_MIN_PURCHASES,
        type=int,
        help='minimum number of purchases',
        dest='min_purchases',
    )
    parser.add_argument(
        '-P', '--products',
        default=DEFAULT_MIN_PRODUCTS,
        type=int,
        help='minimum number of product occurrences',
        dest='min_products',
    )
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        dest='output_dir',
    )
    parser.add_argument(
        'events_data',
        type=argparse.FileType(mode='r'),
    )
    parser.add_argument(
        'purchases_data',
        type=argparse.FileType(mode='r'),
    )
    args = parser.parse_args()
    process_data(
        output_dir=args.output_dir,
        events_data=args.events_data,
        purchases_data=args.purchases_data,
        min_events=args.min_events,
        min_purchases=args.min_purchases,
        min_products=args.min_products,
    )
