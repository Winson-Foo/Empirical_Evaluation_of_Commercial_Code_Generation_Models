DEFAULT_MANIFEST_DIR = './'
DEFAULT_MIN_DURATION = 1
DEFAULT_MAX_DURATION = 15
DEFAULT_NUM_WORKERS = 4
DEFAULT_SAMPLE_RATE = 16000

def add_data_options(parser):
    # Create a group for general data options
    data_group = parser.add_argument_group("General Data Options")

    # Add arguments to the group
    data_group.add_argument('--manifest-dir', default=DEFAULT_MANIFEST_DIR, type=str,
                            help='Output directory for manifests')
    data_group.add_argument('--duration', nargs=2, default=[DEFAULT_MIN_DURATION, DEFAULT_MAX_DURATION], type=int,
                            metavar=('MIN', 'MAX'),
                            help='Prunes training samples shorter than MIN or longer than MAX (in seconds)')
    data_group.add_argument('--num-workers', default=DEFAULT_NUM_WORKERS, type=int,
                            help='Number of workers for processing data.')
    parser.add_argument('--sample-rate', default=DEFAULT_SAMPLE_RATE, type=int, help='Sample rate')

    return parser

# Usage:
# parser = argparse.ArgumentParser()
# parser = add_data_options(parser)
# args = parser.parse_args()