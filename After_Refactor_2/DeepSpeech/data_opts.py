import argparse

def add_data_opts(parser):
    parser.add_argument("--sample-rate", default=16000, type=int, help="Sample rate of the audio.")
    parser.add_argument("--min-duration", default=1, type=int, help="Minimum duration of samples (in seconds).")
    parser.add_argument("--max-duration", default=15, type=int, help="Maximum duration of samples (in seconds).")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers to use for creating manifest.")
    parser.add_argument("--manifest-dir", default="./", type=str, help="Directory to output manifest files.")
    
    return parser