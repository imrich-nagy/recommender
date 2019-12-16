import csv
import sys

from matplotlib import pyplot


pyplot.style.use('ggplot')
pyplot.rc('figure', figsize=(12, 6), dpi=100)

with open(sys.argv[1], 'r') as file:
    reader = csv.DictReader(file)
    log = list(reader)

pyplot.figure()
pyplot.title('Cross-entropy loss')
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.plot(
    [float(row['loss']) for row in log],
    label='Training loss',
)
pyplot.plot(
    [float(row['val_loss']) for row in log],
    label='Validation loss',
)
pyplot.legend()
pyplot.savefig('loss.png')

pyplot.figure()
pyplot.title('Precision')
pyplot.xlabel('Epoch')
pyplot.ylabel('Precision')
pyplot.plot(
    [float(row['precision']) for row in log],
    label='Training precision',
)
pyplot.plot(
    [float(row['val_precision']) for row in log],
    label='Validation precision',
)
pyplot.plot(
    [0.05157593 for row in log],
    label='Baseline precision',
)
pyplot.legend()
pyplot.savefig('precision.png')

pyplot.figure()
pyplot.title('Precision@5')
pyplot.xlabel('Epoch')
pyplot.ylabel('Precision@5')
pyplot.plot(
    [float(row['precision_top_5']) for row in log],
    label='Training precision@5',
)
pyplot.plot(
    [float(row['val_precision_top_5']) for row in log],
    label='Validation precision@5',
)
pyplot.plot(
    [0.04957019 for row in log],
    label='Baseline precision@5',
)
pyplot.legend()
pyplot.savefig('precision-5.png')
