from setuptools import setup, find_packages, find_namespace_packages
import pathlib

path_here = pathlib.Path(__file__).parent.resolve()
long_description = (path_here / 'README.md').read_text(encoding='utf-8')

docs_pkgs = []
tests_pkgs = [
    'pytest',
    'pytest-asyncio',
    'pytest-cov',
]
style_pkgs = [
    "black==22.3.0",  # an in-place reformatter that (mostly) adheres to PEP8
    "flake8==3.9.2",  # sorts and formats import statements inside Python scripts
    "isort==5.10.1",  # a code linter with stylistic conventions that adhere to PEP8
]

setup(
    name='sofamo',
    version='0.0.1',
    license='BSD',
    description='A useful module',
    long_description=long_description,
    platforms=['any'],
    packages=find_packages('src'),
    # packages=find_namespace_packages(where='src', include='package_name*'),
    package_dir={'': 'src'},
    install_requires=[
        "pyaml"
    ],
    extras_require={
        'docs': docs_pkgs,
        'tests': tests_pkgs,
        'dev': docs_pkgs + tests_pkgs + style_pkgs,
    },
)
