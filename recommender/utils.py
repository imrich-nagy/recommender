import argparse


def tqdm_noop(iterable, *args, **kwargs):
    return iterable


class AppendSubset(argparse._AppendAction):

    def __call__(self, parser, namespace, values, option_string=None):
        name, size = values
        try:
            values = (name, float(size))
        except ValueError:
            msg = f'invalid float value: {repr(size)}'
            raise argparse.ArgumentError(self, msg)
        return super(AppendSubset, self).__call__(
            parser=parser,
            namespace=namespace,
            values=values,
            option_string=option_string,
        )
