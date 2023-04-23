# Import the necessary module for executing scrapy.
from scrapy.cmdline import execute


def run_scrapy():
    # Executes the scrapy crawler.
    execute()


if __name__ == "__main__":
    # Call the run_scrapy function.
    run_scrapy()