import argparse
import cProfile
import inspect
import os
import sys
import pkg_resources

import scrapy
from scrapy.commands import BaseRunSpiderCommand, ScrapyCommand, ScrapyHelpFormatter
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.misc import walk_modules
from scrapy.utils.project import get_project_settings, inside_project
from scrapy.utils.python import garbage_collect


class ScrapyArgumentParser(argparse.ArgumentParser):
    """
    Custom argument parser class to ignore any parameters that start with '-:'
    """

    def _parse_optional(self, arg_string):
        if arg_string[:2] == "-:":
            return None
        return super()._parse_optional(arg_string)


def get_command_classes():
    """
    Returns a dictionary of available Scrapy command classes, sourced from the
    'scrapy.commands' submodules and from entry points.
    """
    commands = {}
    for module_name in ["scrapy.commands", get_project_settings().get("COMMANDS_MODULE", "")]:
        for module in walk_modules(module_name):
            for obj in vars(module).values():
                if (inspect.isclass(obj)
                    and issubclass(obj, ScrapyCommand)
                    and obj.__module__ == module.__name__
                    and obj not in (ScrapyCommand, BaseRunSpiderCommand)
                ):
                    commands[obj.__module__.split(".")[-1]] = obj()
    for entry_point in pkg_resources.iter_entry_points("scrapy.commands"):
        if inspect.isclass(entry_point) and issubclass(entry_point, ScrapyCommand):
            commands[entry_point.name] = entry_point()
        else:
            raise Exception(f"Invalid entry point {entry_point.name}")
    return commands


def print_commands():
    """
    Prints a list of available Scrapy commands and their descriptions.
    """
    settings = get_project_settings()
    in_project = inside_project()
    version = scrapy.__version__

    if in_project:
        print(f"Scrapy {version} - active project: {settings['BOT_NAME']}\n")
    else:
        print(f"Scrapy {version} - no active project\n")

    print("Usage:")
    print("  scrapy <command> [options] [args]\n")
    print("Available commands:")
    commands = get_command_classes()
    for command_name, command_class in sorted(commands.items()):
        print(f"  {command_name:<13} {command_class.short_desc()}")

    if not in_project:
        print("\n  [ more ]      More commands available when run from project directory")

    print('\nUse "scrapy <command> -h" to see more info about a command.')


def run_command():
    """
    Parses command line arguments, initializes a CrawlerProcess, and executes the
    specified Scrapy command.
    """
    settings = get_project_settings()
    commands = get_command_classes()
    command_name = sys.argv.pop(1) if len(sys.argv) > 1 else ""

    if not command_name:
        print_commands()
        sys.exit(0)

    if command_name not in commands:
        print(f"Unknown command: {command_name}\n")
        print_commands()
        sys.exit(2)

    command_instance = commands[command_name]
    parser = ScrapyArgumentParser(
        formatter_class=ScrapyHelpFormatter,
        usage=f"scrapy {command_name} {command_instance.syntax()}",
        conflict_handler="resolve",
        description=command_instance.long_desc(),
    )
    settings.setdict(command_instance.default_settings, priority="command")
    command_instance.settings = settings
    command_instance.add_options(parser)
    opts, args = parser.parse_known_args()

    try:
        command_instance.process_options(args, opts)
    except UsageError as e:
        if str(e):
            parser.error(str(e))
        if e.print_help:
            parser.print_help()
        sys.exit(2)

    command_instance.crawler_process = CrawlerProcess(settings)
    if opts.profile:
        run_command_profiled(command_instance, args, opts)
    else:
        command_instance.run(args, opts)
    sys.exit(command_instance.exitcode)


def run_command_profiled(command_instance, args, opts):
    """
    Runs a Scrapy command with profiling enabled.
    """
    if opts.profile:
        sys.stderr.write(f"scrapy: writing cProfile stats to {opts.profile!r}\n")
    loc = locals()
    p = cProfile.Profile()
    p.runctx("command_instance.run(args, opts)", globals(), loc)
    if opts.profile:
        p.dump_stats(opts.profile)


if __name__ == "__main__":
    try:
        run_command()
    finally:
        garbage_collect() 