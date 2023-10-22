DEFAULT_MANIFEST_DIR = './'
DEFAULT_MIN_DURATION = 1
DEFAULT_MAX_DURATION = 15
DEFAULT_NUM_WORKERS = 4
DEFAULT_SAMPLE_RATE = 16000

MANIFEST_DIR_HELP = 'Output directory for manifests'
MIN_DURATION_HELP = 'Prunes training samples shorter than the min duration (given in seconds, default 1)'
MAX_DURATION_HELP = 'Prunes training samples longer than the max duration (given in seconds, default 15)'
NUM_WORKERS_HELP = 'Number of workers for processing data.'
SAMPLE_RATE_HELP = 'Sample rate'


def create_data_opts(parser):
    data_opts = parser.add_argument_group("General Data Options")
    data_opts.add_argument('--manifest-dir', type=str, default=DEFAULT_MANIFEST_DIR, help=MANIFEST_DIR_HELP)
    data_opts.add_argument('--min-duration', type=int, default=DEFAULT_MIN_DURATION, help=MIN_DURATION_HELP)
    data_opts.add_argument('--max-duration', type=int, default=DEFAULT_MAX_DURATION, help=MAX_DURATION_HELP)
    return data_opts


def add_data_opts(parser):
    data_opts = create_data_opts(parser)
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS, help=NUM_WORKERS_HELP)
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE, help=SAMPLE_RATE_HELP)
    return parser
