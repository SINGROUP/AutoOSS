from setuptools import setup, find_packages
from os.path import join, dirname

requirementstxt = join(dirname(__file__), "requirements.txt")
requirements = [ line.strip() for line in open(requirementstxt, "r") if line.strip() ]

setup(
    name="AutoOSS",
    version="0.1.0",
    author="Nian Wu",
    author_email="wunianwhu@gmail.com",
    description="Automate chemical reactions (breaking C-Br bond) in STM",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # Optional: if you have a README.md file
    url="https://github.com/Meganwu/AutoOSS",
    packages=find_packages(),  # Automatically find packages in your source directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: Automation :: Chemical synthesis",
    ],

    # To remotely monitor STM, you need to install pywin32.
    extras_require={
        'run': [
            'pywin32'
        ]
    },
    install_requires=requirements,
)
