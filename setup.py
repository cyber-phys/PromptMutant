from setuptools import setup, find_packages

setup(
    name='promptmutant',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    url='https://github.com/cyber-phys/promptmutant',
    license='MIT',
    author='Luc Chartier',
    author_email='luc@cyber-phys.com',
    description='A basic implementation of the Promptbreeder system',
    entry_points={
    "console_scripts": [
        "promptmutant = promptmutant.__main__:main"
    ]
    }
)

