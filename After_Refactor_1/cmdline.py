import argparse
import cProfile
import inspect
import os
import sys
from typing import Dict, Any, List

import pkg_resources

import scrapy
from scrapy.commands import BaseRunSpiderCommand, ScrapyCommand
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import UsageError
from scrapy.utils.misc import walk_modules
from scrapy.utils.project import get_project_settings, inside_project
from scrapy.utils.python import garbage_collect


class ScrapyArgumentParser(argparse.ArgumentParser):
    def _parse_optional(self, arg_string: str) -> Any:
        """
        Override the optional argument parser to skip over parameter arguments 
        that begin with "-:".

        :param arg_string: The argument to parse.
        :return: The parsed argument, or None if the argument should be skipped.
        """
        if arg_string[:2] == "-:":
            return None
        return super()._parse_optional(arg_string)


def get_commands_dict(settings: Dict[str, Any], in_project: bool) -> Dict[str, ScrapyCommand]:
    """
    Get the available commands for the given scrapy settings.

    :param settings: The scrapy settings object to get commands for.
    :param in_project: Whether the current directory is inside a scrapy project.
    :return: A dictionary of the available commands, where the keys are command names
        and the values are command objects.
    """
    all_commands = {}
    for cmd_class in walk_modules("scrapy.commands"):
        if (
            inspect.isclass(cmd_class) 
            and issubclass(cmd_class, ScrapyCommand) 
            and cmd_class.__module__ == "scrapy.commands" 
            and cmd_class not in (ScrapyCommand, BaseRunSpiderCommand)
        ):
            cmd_name = cmd_class.__module__.split(".")[-1]
            cmd = cmd_class()
            if in_project or not cmd.requires_project:
                all_commands[cmd_name] = cmd
    entry_points = {
        ep.name: ep.load() 
        for ep in pkg_resources.iter_entry_points("scrapy.commands")
    }
    all_commands.update(entry_points)
    commands_module = settings["COMMANDS_MODULE"]
    if commands_module:
        module_commands = _get_commands_from_module(commands_module, in_project)
        all_commands.update(module_commands)
    return all_commands


def _pop_command_name(argv: List[str]) -> str:
    """
    Pop the command name from the argument list.

    :param argv: The list of arguments.
    :return: The command name, or an empty string if no command name was found.
    """
    for i, arg in enumerate(argv[1:], start=1):
        if not arg.startswith("-"):
            del argv[i - 1]
            return arg
    return ""


def _print_commands(settings: Dict[str, Any], in_project: bool) -> None:
    """
    Print the list of available scrapy commands to the console.

    :param settings: The current scrapy settings.
    :param in_project: Whether the current directory is inside a scrapy project.
    """
    version = scrapy.__version__
    project_name = settings.get("BOT_NAME", "no active project")
    if in_project:
        header = f"Scrapy {version} - active project: {project_name}\n"
    else:
        header = f"Scrapy {version} - no active project\n"
    print(header)
    print("Usage:")
    print("  scrapy <command> [options] [args]\n")
    print("Available commands:")
    commands = get_commands_dict(settings, in_project)
    for cmd_name, cmd in sorted(commands.items()):
        print(f"  {cmd_name:<13} {cmd.short_desc()}")
    if not in_project:
        print()
        print("  [ more ]      More commands available when run from project directory")
    print()
    print('Use "scrapy <command> -h" to see more info about a command')


def _print_unknown_command(settings: Dict[str, Any], cmd_name: str, in_project: bool) -> None:
    """
    Print an error message when the specified scrapy command does not exist.

    :param settings: The current scrapy settings.
    :param cmd_name: The name of the unknown command.
    :param in_project: Whether the current directory is inside a scrapy project.
    """
    version = scrapy.__version__
    project_name = settings.get("BOT_NAME", "no active project")
    if in_project:
        header = f"Scrapy {version} - active project: {project_name}\n"
    else:
        header = f"Scrapy {version} - no active project\n"
    print(header)
    print(f"Unknown command: {cmd_name}\n")
    print('Use "scrapy" to see available commands')


def execute(argv: List[str] = None, settings: Dict[str, Any] = None) -> None:
    """
    Parse the command line arguments and execute the specified scrapy command.

    :param argv: The list of command line arguments.
    :param settings: The scrapy settings to use for the command.
    """
    if argv is None:
        argv = sys.argv
    if settings is None:
        settings = get_project_settings()
        # set EDITOR from environment if available
        editor = os.environ.get("EDITOR")
        if editor:
            settings["EDITOR"] = editor
    in_project = inside_project()
    cmd_name = _pop_command_name(argv)
    if not cmd_name:
        _print_commands(settings, in_project)
        sys.exit(0)
    commands = get_commands_dict(settings, in_project)
    if cmd_name not in commands:
        _print_unknown_command(settings, cmd_name, in_project)
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
    try:
        cmd.process_options(args, opts)
        cmd.crawler_process = CrawlerProcess(settings)
        if opts.profile:
            _run_command_profiled(cmd, args, opts)
        else:
            cmd.run(args, opts)
    except UsageError as e:
        parser.error(str(e))
        sys.exit(2)


def _run_command_profiled(cmd: ScrapyCommand, args: List[str], opts: argparse.Namespace) -> None:
    """
    Run the specified scrapy command with profiling enabled.

    :param cmd: The scrapy command to run.
    :param args: The list of command arguments.
    :param opts: The command line options.
    """
    if opts.profile:
        sys.stderr.write(f"scrapy: writing cProfile stats to {opts.profile!r}\n")
    loc = locals()
    p = cProfile.Profile()
    p.runctx("cmd.run(args, opts)", globals(), loc)
    if opts.profile:
        p.dump_stats(opts.profile)


if __name__ == "__main__":
    try:
        execute()
    finally:
        garbage_collect()