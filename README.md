# Fashion Similarities
This package implements a convolutional autoencoder to construct low dimensional representations of fashion article images. These low dimensional representations can then be used to compute similarity scores between different articles. For fast comparison, this work also implements LSH to hash the latent representation into buckets, which determine candidates for comparison.

The architecture of the final convolutional autoencoder is as follow:

![CAE Architecture](https://github.com/LTluttmann/fashion_similarities/blob/master/architecture.png?raw=true)
