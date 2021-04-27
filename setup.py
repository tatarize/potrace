from setuptools import setup
setup(
    install_requires=[
        "numpy",
    ],
    extras_require={
        'cli': ["potrace-cli"],
    }
)