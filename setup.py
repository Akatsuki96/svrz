from setuptools import setup, find_packages

def get_requirements():
    with open("./requirements.txt", 'r') as f:
        requirements = f.readlines()
    print(requirements)
    return requirements

setup(
	name="ssvr",
	version="0.0.1",
	author="Akatsuki96",
	author_email="marco.rando0396@gmail.com",
	packages=find_packages(),
	install_requires=get_requirements(),
	python_requires=">=3.9",
)