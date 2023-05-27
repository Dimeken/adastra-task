from setuptools import setup

setup(
    name='movie_analyzer',
    version='1.0',
    packages=['movie_analyzer'],
    entry_points={
        'console_scripts': [
            'movie_analyzer = movie_analyzer.main:main'
        ]
    },
    install_requires=[
        'pandas',
        'jsonschema'
    ],
)
