from setuptools import setup

setup(
    name='FashionSimilarities',
    version='0.5.0',
    author='Laurin Luttmann',
    packages=['fashion_similarities', 'fashion_similarities.test'],
    license='LICENSE.txt',
    description='Implements Autoencoder and LSH for retrieving similar fashion images',
    long_description=open('README.txt').read(),
    install_requires=[
        "tensorflow",
        "numpy",
        "keras",
        "scikit-learn",
        "joblib",
        "pillow"
    ],
)