# Assignment 3, Part 1: Variational Autoencoders

This folder contains the code for implementing your own VAE model.
We will train the model on generating 4-bit MNIST images. The original MNIST dataset contains images with pixel values between 0 and 1. To discretize those, we multiply pixel values with 16 and map the result to the closest integer value (rounding down 16 to 15). This is a 4-bit representation of the original image. Standard RGB images are usually using 8-bit encodings (i.e. values between 0 and 255), but to simplify the task, we only use 4 bits here.

The code is structured in the following way:
* `fmnist.py`: Contains a function for preparing the discretized dataset and providing a data loader for training, validation and testing.
* `cnn_encoder_decoder.py`: Contains template classes for the Encoder and Decoder based on an CNN.
* `train_pl.py`: Contains training functionalities such as the training loop, logging, saving, etc. We have provided you with logging utilities and general code structure so that you can focus on the important parts of the VAE model.
* `utils.py`: Contains functionalities that are required for training, such as the reparameterization trick, the KL divergence, bpd calculation and manifold generation.
* `unittests.py`: Contains unittests for the Encoder and Decoder networks, as well as functions of `utils.py`. It will hopefully help you debugging your code. Your final code should pass these unittests.

Default hyperparameters are provided in the `ArgumentParser` object of the respective training functions. The model should be able to generate decent images with the default hyperparameters.
  If you test the code on your local machine, you can use the argument `--progress_bar` to show a training progressbar. Remember to not use this on Lisa as it otherwise fills up your SLURM output file very quickly. It is recommended to look at the TensorBoard there instead.
  The training time with the default hyperparameters is less than 15 minutes on a NVIDIA GTX1080Ti (GPU provided on Lisa).

