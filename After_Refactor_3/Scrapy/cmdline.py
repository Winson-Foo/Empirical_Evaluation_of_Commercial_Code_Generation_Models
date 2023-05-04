def iter_command_classes(module_name):
    # Gets all classes in a module that are subclasses of scrapy.commands.ScrapyCommand
    # and returns them as a generator.
    for module in walk_modules(module_name):
        for obj in vars(module).values():
            if (
                    inspect.isclass(obj)
                    and issubclass(obj, ScrapyCommand)
                    and obj.__module__ == module.__name__
                    and obj not in (ScrapyCommand, BaseRunSpiderCommand)
            ):
                yield obj

def get_commands_from_module(module_name, inproject):
    # Given a module name and a boolean value for inproject, returns a dictionary of all commands
    # defined in that module that can be run either inside or outside of a scrapy project.
    command_dict = {}
    for command in iter_command_classes(module_name):
        if inproject or not command.requires_project:
            command_name = command.__module__.split(".")[-1]
            command_dict[command_name] = command()
    return command_dict
    
def create_parser(command):
    # Given a command object, creates an argparse.ArgumentParser object
    # populated with the command's default settings
    parser = ScrapyArgumentParser(
        formatter_class=ScrapyHelpFormatter,
        usage=f"scrapy {command_name} {command.syntax()}",
        conflict_handler="resolve",
        description=command.long_desc(),
    )
    command.add_options(parser)  # assumes command is populated with default settings in get_commands() function
    return parser 
    
def parse_args(parser, argv):
    # Given a argparse.ArgumentParser object `parser` and a list of arguments `argv`,
    # returns a tuple containing the parsed options and arguments.
    return parser.parse_known_args(args=argv[1:])

def handle_options(command, parser, opts, args):
    # Populates command object with settings, default_settings, and crawler_process
    # based on parsed options, parser, and arguments.
    settings = command.settings 
    settings.setdict(command.default_settings, priority="command")
    command.crawler_process = CrawlerProcess(settings)
    
    # If --profile flag is used, execute command with profiler
    if opts.profile:
        _run_command_profiled(command, args, opts)

def execute_command(command, settings, opts, args):
    # Executes a command with parsed options and arguments.
    if opts.profile:
        _run_command_profiled(command, args, opts)
    else:
        command.run(args, opts)