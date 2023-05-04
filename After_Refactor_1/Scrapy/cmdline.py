import argparse
import cProfile
import inspect
import os
import sys

import pkg_resources

import scrapy
from scrapy.exceptions import UsageError
from scrapy.utils.misc import walk_modules
from scrapy.utils.project import get_project_settings, inside_project
from scrapy.commands import BaseRunSpiderCommand, ScrapyCommand, ScrapyHelpFormatter
from scrapy.crawler import CrawlerProcess


def is_parameter(arg_string: str) -> bool:
    """Returns True if the argument is a parameter, not an argument"""
    return arg_string[:2] == "-:"


class ScrapyArgumentParser(argparse.ArgumentParser):
    def _parse_optional(self, arg_string):
        return None if is_parameter(arg_string) else super()._parse_optional(arg_string)


def iter_command_classes(module_name: str):
    """Iterates over all Scrapy command classes defined in the given module"""
    for module in walk_modules(module_name):
        for obj in vars(module).values():
            if (inspect.isclass(obj)
                    and issubclass(obj, ScrapyCommand)
                    and obj.__module__ == module.__name__
                    and obj not in (ScrapyCommand, BaseRunSpiderCommand)):
                yield obj


def get_commands_from_module(module: str, in_project: bool):
    """Returns a dictionary of commands defined in the given module"""
    commands = {}
    for cmd in iter_command_classes(module):
        if in_project or not cmd.requires_project:
            cmd_name = cmd.__module__.split(".")[-1]
            commands[cmd_name] = cmd()
    return commands


def get_commands_from_entry_points(in_project: bool, group: str = "scrapy.commands"):
    """Returns a dictionary of commands defined in entry points"""
    commands = {}
    for entry_point in pkg_resources.iter_entry_points(group):
        obj = entry_point.load()
        if inspect.isclass(obj):
            commands[entry_point.name] = obj()
        else:
            raise Exception(f"Invalid entry point {entry_point.name}")
    return commands


def get_commands_dict(settings, in_project):
    """Returns a dictionary of all available Scrapy commands"""
    commands = get_commands_from_module("scrapy.commands", in_project)
    commands.update(get_commands_from_entry_points(in_project))
    cmds_module = settings["COMMANDS_MODULE"]
    if cmds_module:
        commands.update(get_commands_from_module(cmds_module, in_project))
    return commands


def pop_command_name(argv):
    """Pops the command name from the argument list"""
    i = 0
    for arg in argv[1:]:
        if not arg.startswith("-"):
            del argv[i]
            return arg
        i += 1


def print_header(settings, in_project):
    """Prints the Scrapy version and active project (if any)"""
    version = scrapy.__version__
    if in_project:
        print(f"Scrapy {version} - active project: {settings['BOT_NAME']}\n")
    else:
        print(f"Scrapy {version} - no active project\n")


def print_commands(settings, in_project):
    """Prints a list of available Scrapy commands"""
    print_header(settings, in_project)
    print("Usage:")
    print("  scrapy <command> [options] [args]\n")
    print("Available commands:")
    commands = get_commands_dict(settings, in_project)
    for cmd_name, cmd_class in sorted(commands.items()):
        print(f"  {cmd_name:<13} {cmd_class.short_desc()}")
    if not in_project:
        print()
        print("  [ more ]      More commands available when run from project directory")
    print()
    print('Use "scrapy <command> -h" to see more info about a command')


def print_unknown_command(settings, cmd_name, in_project):
    """Prints an error message for an unknown command"""
    print_header(settings, in_project)
    print(f"Unknown command: {cmd_name}\n")
    print('Use "scrapy" to see available commands')


def run_print_help(parser, func, *a, **kw):
    """Runs a function while catching UsageError and printing help if necessary"""
    try:
        func(*a, **kw)
    except UsageError as e:
        if str(e):
            parser.error(str(e))
        if e.print_help:
            parser.print_help()
        sys.exit(2)


def run_command(cmd, args, opts):
    """Runs a Scrapy command"""
    if opts.profile:
        run_command_profiled(cmd, args, opts)
    else:
        cmd.run(args, opts)


def run_command_profiled(cmd, args, opts):
    """Runs a Scrapy command under cProfile"""
    if opts.profile:
        sys.stderr.write(f"scrapy: writing cProfile stats to {opts.profile!r}\n")
    loc = locals()
    p = cProfile.Profile()
    p.runctx("cmd.run(args, opts)", globals(), loc)
    if opts.profile:
        p.dump_stats(opts.profile)


def execute(argv=None, settings=None):
    """The main entry point for running Scrapy commands"""
    if argv is None:
        argv = sys.argv

    if settings is None:
        settings = get_project_settings()
        # set EDITOR from environment if available
        try:
            editor = os.environ["EDITOR"]
        except KeyError:
            pass
        else:
            settings["EDITOR"] = editor

    in_project = inside_project()
    commands = get_commands_dict(settings, in_project)
    cmd_name = pop_command_name(argv)
    if not cmd_name:
        print_commands(settings, in_project)
        sys.exit(0)
    elif cmd_name not in commands:
        print_unknown_command(settings, cmd_name, in_project)
        sys.exit(2)

    cmd = commands[cmd_name]
    parser = ScrapyArgumentParser(
        formatter_class=ScrapyHelpFormatter,
        usage=f"scrapy {cmd_name} {cmd.syntax()}",
        conflict_handler="resolve",
        description=cmd.long_desc(),
    )
    settings.setdict(cmd.default_settings, priority="command")
    cmd.settings = settings
    cmd.add_options(parser)
    opts, args = parser.parse_known_args(args=argv[1:])
    run_print_help(parser, cmd.process_options, args, opts)

    cmd.crawler_process = CrawlerProcess(settings)
    run_print_help(parser, run_command, cmd, args, opts)
    sys.exit(cmd.exitcode)