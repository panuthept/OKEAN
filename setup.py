from setuptools import setup, find_packages


# with open("requirements.txt") as f:
#     dependencies = [line for line in f]

setup(
    name='ocean',
    version='0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache-2.0 License',
    author='Panuthep Tasawong',
    author_email='panuthep.t_s20@vistec.ac.th',
    description='A Universal Framework for Knowledge-Enhanced Information Retrieval (KEIR) Applications',
    python_requires='>=3.11',
    # install_requires=dependencies
)