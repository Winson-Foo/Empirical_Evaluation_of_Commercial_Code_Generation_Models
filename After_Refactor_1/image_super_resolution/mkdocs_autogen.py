import ast
import os
import re
from typing import Any, Dict, List, Optional


def delete_space(parts: List[str], start: int, end: int) -> Optional[str]:
    """
    Deletes the spaces surrounding the lines of code.

    Args:
        parts: A List of strings containing the code.
        start: An integer containing the starting index of the line from where the space needs to be deleted.
        end: An integer containing the ending index of the line until where the space needs to be deleted.

    Returns:
        A string containing the code.

    """
    if start > end or end >= len(parts):
        return None
    count = 0
    while count < len(parts[start]):
        if parts[start][count] == ' ':
            count += 1
        else:
            break
    return '\n'.join(y for y in [x[count:] for x in parts[start: end + 1]] if len(x) > count)


def change_args_to_dict(string: str) -> Optional[Dict[str, Any]]:
    """
    Changes the arguments to a dictionary.

    Args:
        string: A string containing the function arguments.

    Returns:
        A dictionary containing the arguments.

    """
    if string is None:
        return None
    ans = []
    strings = string.split('\n')
    ind = 1
    start = 0
    while ind <= len(strings):
        if ind < len(strings) and strings[ind].startswith(" "):
            ind += 1
        else:
            if start < ind:
                ans.append('\n'.join(strings[start:ind]))
            start = ind
            ind += 1
    d = {}
    for line in ans:
        if ":" in line and len(line) > 0:
            lines = line.split(":")
            d[lines[0]] = lines[1].strip()
    return d


def remove_next_line(comments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes new lines from the comments section.

    Args:
        comments: A dictionary containing comments.

    Returns:
        A dictionary containing comments without new lines.

    """
    for x in comments:
        if comments[x] is not None and '\n' in comments[x]:
            comments[x] = ' '.join(comments[x].split('\n'))
    return comments


def skip_space_line(parts: List[str], ind: int) -> int:
    """
    Skips the lines with spaces.

    Args:
        parts: A List of strings containing the code.
        ind: An integer containing the index of the line to start from.

    Returns:
        An integer containing the index of the next available line with code.

    """
    while ind < len(parts):
        if re.match(r'^\s*$', parts[ind]):
            ind += 1
        else:
            break
    return ind


def parse_func_string(comment: Optional[str]) -> Dict[str, Any]:
    """
    Parses the function comments string and extracts relevant information.

    Args:
        comment: A string containing the comments for function.

    Returns:
        A dictionary containing the comments of the function.

    """
    if comment is None or len(comment) == 0:
        return {}
    comments = {}
    paras = ('Args', 'Attributes', 'Methods', 'Returns', 'Raises')
    comment_parts = [
        'short_description',
        'long_description',
        'Args',
        'Attributes',
        'Methods',
        'Returns',
        'Raises',
    ]
    for x in comment_parts:
        comments[x] = None

    parts = re.split(r'\n', comment)
    ind = 1
    while ind < len(parts):
        if re.match(r'^\s*$', parts[ind]):
            break
        else:
            ind += 1

    comments['short_description'] = '\n'.join(
        ['\n'.join(re.split('\n\s+', x.strip())) for x in parts[0:ind]]
    ).strip(':\n\t ')
    ind = skip_space_line(parts, ind)

    start = ind
    while ind < len(parts):
        if parts[ind].strip().startswith(paras):
            break
        else:
            ind += 1
    long_description = '\n'.join(
        ['\n'.join(re.split('\n\s+', x.strip())) for x in parts[start:ind]]
    ).strip(':\n\t ')
    comments['long_description'] = long_description

    ind = skip_space_line(parts, ind)
    while ind < len(parts):
        if parts[ind].strip().startswith(paras):
            start = ind
            start_with = parts[ind].strip()
            ind += 1
            while ind < len(parts):
                if parts[ind].strip().startswith(paras):
                    break
                else:
                    ind += 1
            part = delete_space(parts, start + 1, ind - 1)
            if start_with.startswith(paras[0]):
                comments[paras[0]] = change_args_to_dict(part)
            elif start_with.startswith(paras[1]):
                comments[paras[1]] = change_args_to_dict(part)
            elif start_with.startswith(paras[2]):
                comments[paras[2]] = change_args_to_dict(part)
            elif start_with.startswith(paras[3]):
                comments[paras[3]] = change_args_to_dict(part)
            elif start_with.startswith(paras[4]):
                comments[paras[4]] = part
            ind = skip_space_line(parts, ind)
        else:
            ind += 1

    remove_next_line(comments)
    return comments


def md_parse_line_break(comment: str) -> str:
    """
    Parses the markdown line break.

    Args:
        comment: A string containing the markdown comment.

    Returns:
        A formatted string.

    """
    comment = comment.replace('  ', '\n\n')
    return comment.replace(' - ', '\n\n- ')


def to_md(comment_dict: Dict[str, Any]) -> str:
    """
    Converts the comments dictionary to a markdown formatted string.

    Args:
        comment_dict: A dictionary containing the comments.

    Returns:
        A markdown formatted string containing the comments.

    """
    doc = ''
    if 'short_description' in comment_dict:
        doc += comment_dict['short_description']
        doc += '\n\n'

    if 'long_description' in comment_dict:
        doc += md_parse_line_break(comment_dict['long_description'])
        doc += '\n'

    if 'Args' in comment_dict and comment_dict['Args'] is not None:
        doc += '##### Args\n'
        for arg, des in comment_dict['Args'].items():
            doc += '* **' + arg + '**: ' + des + '\n\n'

    if 'Attributes' in comment_dict and comment_dict['Attributes'] is not None:
        doc += '##### Attributes\n'
        for arg, des in comment_dict['Attributes'].items():
            doc += '* **' + arg + '**: ' + des + '\n\n'

    if 'Methods' in comment_dict and comment_dict['Methods'] is not None:
        doc += '##### Methods\n'
        for arg, des in comment_dict['Methods'].items():
            doc += '* **' + arg + '**: ' + des + '\n\n'

    if 'Returns' in comment_dict and comment_dict['Returns'] is not None:
        doc += '##### Returns\n'
        if isinstance(comment_dict['Returns'], str):
            doc += comment_dict['Returns']
            doc += '\n'
        else:
            for arg, des in comment_dict['Returns'].items():
                doc += '* **' + arg + '**: ' + des + '\n\n'
    return doc


def parse_func_args(function: Any) -> str:
    """
    Parses the function arguments.

    Args:
        function: A Function object.

    Returns:
        A string containing the function arguments.

    """
    args = [a.arg for a in function.args.args if a.arg != 'self']
    kwargs = []
    if function.args.kwarg:
        kwargs = ['**' + function.args.kwarg.arg]

    return '(' + ', '.join(args + kwargs) + ')'


def get_func_comments(function_definitions: List[Any]) -> str:
    """
    Extracts the function comments.

    Args:
        function_definitions: A List of function objects.

    Returns:
        A string containing the function comments.

    """
    doc = ''
    for f in function_definitions:
        temp_str = to_md(parse_func_string(ast.get_docstring(f)))
        doc += ''.join(
            [
                '### ',
                f.name.replace('_', '\\_'),
                '\n',
                '```python',
                '\n',
                'def ',
                f.name,
                parse_func_args(f),
                '\n',
                '```',
                '\n',
                temp_str,
                '\n',
            ]
        )

    return doc


def get_comments_str(file_name: str) -> str:
    """
    Extracts comments from the given Python file.

    Args:
        file_name: A string containing the file path.

    Returns:
        A string containing the comments.

    """
    with open(file_name) as fd:
        file_contents = fd.read()
    module = ast.parse(file_contents)

    function_definitions = [node for node in module.body if
                            isinstance(node, ast.FunctionDef) and (node.name[0] != '_' or node.name[:2] == '__')]

    doc = get_func_comments(function_definitions)

    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for class_def in class_definitions:
        temp_str = to_md(parse_func_string(ast.get_docstring(class_def)))

        # excludes private methods (start with '_')
        method_definitions = [
            node
            for node in class_def.body
            if isinstance(node, ast.FunctionDef) and (node.name[0] != '_' or node.name[:2] == '__')
        ]

        temp_str += get_func_comments(method_definitions)
        doc += '## class ' + class_def.name + '\n' + temp_str
    return doc
