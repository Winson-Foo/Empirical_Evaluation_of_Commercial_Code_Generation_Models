import ast
import os
import re


def delete_indentation(parts, start, end):
    """
    Helper function to delete indentation from parts of a string.
    """
    if start > end or end >= len(parts):
        return None
    count = 0
    while count < len(parts[start]):
        if parts[start][count] == ' ':
            count += 1
        else:
            break
    return '\n'.join(y for y in [x[count:] for x in parts[start: end + 1] if len(x) > count])


def string_to_dict(string):
    """
    Helper function to convert a string of arguments to a dictionary.
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


def remove_newlines(comments):
    """
    Helper function to remove all newlines from a dictionary of comments.
    """
    for x in comments:
        if comments[x] is not None and '\n' in comments[x]:
            comments[x] = ' '.join(comments[x].split('\n'))
    return comments


def parse_func_string(comment):
    """
    Parse the docstring of a function into a dictionary of comments.
    """
    if comment is None or len(comment) == 0:
        return {}
    
    comments = {
        'short_description': None,
        'long_description': None,
        'Args': None,
        'Attributes': None,
        'Methods': None,
        'Returns': None,
        'Raises': None
    }
    
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
    
    ind = skip_whitespace(parts, ind)
    
    start = ind
    while ind < len(parts):
        if any(parts[ind].strip().startswith(x) for x in ('Args', 'Attributes', 'Methods', 'Returns', 'Raises')):
            break
        else:
            ind += 1
    
    long_description = '\n'.join(
        ['\n'.join(re.split('\n\s+', x.strip())) for x in parts[start:ind]]
    ).strip(':\n\t ')
    comments['long_description'] = long_description
    
    ind = skip_whitespace(parts, ind)
    while ind < len(parts):
        if any(parts[ind].strip().startswith(x) for x in ('Args', 'Attributes', 'Methods', 'Returns', 'Raises')):
            start = ind
            start_with = parts[ind].strip()
            ind += 1
            while ind < len(parts):
                if any(parts[ind].strip().startswith(x) for x in ('Args', 'Attributes', 'Methods', 'Returns', 'Raises')):
                    break
                else:
                    ind += 1
            part = delete_indentation(parts, start + 1, ind - 1)
            if start_with.startswith('Args'):
                comments['Args'] = string_to_dict(part)
            elif start_with.startswith('Attributes'):
                comments['Attributes'] = string_to_dict(part)
            elif start_with.startswith('Methods'):
                comments['Methods'] = string_to_dict(part)
            elif start_with.startswith('Returns'):
                comments['Returns'] = string_to_dict(part)
            elif start_with.startswith('Raises'):
                comments['Raises'] = part
            ind = skip_whitespace(parts, ind)
        else:
            ind += 1
    
    remove_newlines(comments)
    return comments


def parse_func_args(function):
    """
    Get the function arguments as a string.
    """
    args = [a.arg for a in function.args.args if a.arg != 'self']
    kwargs = []
    if function.args.kwarg:
        kwargs = ['**' + function.args.kwarg.arg]
    
    return '(' + ', '.join(args + kwargs) + ')'


def get_func_comments(function_definitions):
    """
    Parse comments for all functions in the module.
    """
    doc = ''
    for f in function_definitions:
        func_str = parse_func_string(ast.get_docstring(f))
        temp_str = to_markdown(func_str)
        doc += ''.join([
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
                '\n'
        ])
    return doc


def get_class_comments(class_definitions):
    """
    Parse comments for all classes in the module.
    """
    doc = ''
    for c in class_definitions:
        class_str = parse_func_string(ast.get_docstring(c))
        temp_str = to_markdown(class_str)
        
        # excludes private methods (start with '_')
        method_definitions = [node for node in c.body if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
        
        temp_str += get_func_comments(method_definitions)
        doc += '## class ' + c.name + '\n' + temp_str
    return doc


def to_markdown(comment_dict):
    """
    Convert a dictionary of comments to markdown format.
    """
    md_str = ''
    
    if 'short_description' in comment_dict:
        md_str += comment_dict['short_description'] + '\n\n'
    
    if 'long_description' in comment_dict:
        md_str += parse_line_break(comment_dict['long_description']) + '\n'
    
    if 'Args' in comment_dict and comment_dict['Args'] is not None:
        md_str += '##### Args\n'
        for arg, des in comment_dict['Args'].items():
            md_str += f'* **{arg}**: {des}\n\n'
    
    if 'Attributes' in comment_dict and comment_dict['Attributes'] is not None:
        md_str += '##### Attributes\n'
        for arg, des in comment_dict['Attributes'].items():
            md_str += f'* **{arg}**: {des}\n\n'
    
    if 'Methods' in comment_dict and comment_dict['Methods'] is not None:
        md_str += '##### Methods\n'
        for arg, des in comment_dict['Methods'].items():
            md_str += f'* **{arg}**: {des}\n\n'
    
    if 'Returns' in comment_dict and comment_dict['Returns'] is not None:
        md_str += '##### Returns\n'
        if isinstance(comment_dict['Returns'], str):
            md_str += comment_dict['Returns'] + '\n'
        else:
            for arg, des in comment_dict['Returns'].items():
                md_str += f'* **{arg}**: {des}\n\n'
    return md_str


def parse_line_break(comment):
    """
    Convert double spaces to line break in a string.
    """
    comment = comment.replace('  ', '\n\n')
    return comment.replace(' - ', '\n\n- ')


def skip_whitespace(parts, ind):
    """
    Skip over any whitespace lines in a list of strings.
    """
    while ind < len(parts):
        if re.match(r'^\s*$', parts[ind]):
            ind += 1
        else:
            break
    return ind


def extract_comments(directory):
    """
    Extract comments from all python files in a directory and write them to respective markdown files.
    """
    for parent, dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            if os.path.splitext(file_name)[1] == '.py' and file_name != '__init__.py':
                file_contents = read_file(os.path.join(parent, file_name))
                module = ast.parse(file_contents)
                function_definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
                class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
                
                func_comments = get_func_comments(function_definitions)
                class_comments = get_class_comments(class_definitions)
                doc = func_comments + class_comments
                
                directory_out = os.path.join('docs', parent.replace(directory, ''))
                if not os.path.exists(directory_out):
                    os.makedirs(directory_out)
                
                output_file = open(os.path.join(directory_out, file_name[:-3] + '.md'), 'w')
                output_file.write(doc)
                output_file.close()


def read_file(file_path):
    """
    Read contents of a file and return as string.
    """
    with open(file_path) as fd:
        return fd.read()