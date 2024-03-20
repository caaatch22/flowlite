from setuptools import setup, find_packages

setup(
    name='flowlite',
    version='0.0.1',
    author='MJ',
    author_email='mr.liumingjie@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'typing_extension',
    ],
    description='A simple, educational deep learning framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/caaatch22/flowlite',
    # ...更多选项
)
