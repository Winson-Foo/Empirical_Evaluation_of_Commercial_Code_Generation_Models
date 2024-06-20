from os.path import dirname, join

from setuptools import find_packages, setup

def get_version():
    with open(join(dirname(__file__), "foolbox/VERSION")) as f:
        return f.read().strip()

def get_long_description():
    try:
        # obtain long description from README
        readme_path = join(dirname(__file__), "README.rst")
        with open(readme_path, encoding="utf-8") as f:
            README = f.read()
            # remove raw html not supported by PyPI
            README = "\n".join(README.split("\n")[3:])
    except IOError:
        README = ""
    
    return README

def setup_package():
    install_requires = read_configuration("install_requires.txt")
    tests_require = read_configuration("tests_require.txt")
    classifiers = read_configuration("classifiers.txt")
    
    setup(
        name="foolbox",
        version=get_version(),
        description="Foolbox is an adversarial attacks library that works natively with PyTorch, TensorFlow and JAX",
        long_description=get_long_description(),
        long_description_content_type="text/x-rst",
        classifiers=classifiers,
        keywords="",
        author="Jonas Rauber, Roland S. Zimmermann",
        author_email="foolbox+rzrolandzimmermann@gmail.com",
        url="https://github.com/bethgelab/foolbox",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=install_requires,
        extras_require={"testing": tests_require},
    )

if __name__ == "__main__":
    setup_package()