# Neural Networks Coursework

Homework for the BMSTU IU9 neural networks course, 2024.
C++20 implementations of perceptrons and optimization algorithms, plus
convolutional-network experiments in PyTorch.

| Work | Contents |
|---|---|
| [hw1](hw1/) | Recognition of 4×5 symbols; activation functions and backpropagation |
| [hw2](hw2/) | Multilayer perceptron for MNIST; MSE, cross-entropy and KL-divergence |
| [hw3](hw3/) | Rosenbrock-function minimization: gradient descent, Fletcher–Reeves, Polak–Ribière, DFP and damped Newton (Levenberg–Marquardt) |
| [hw4](hw4/) | SGD, Nesterov momentum, Adagrad and Adam; genetic search for network hyperparameters |
| [hw5](hw5/) | LeNet-5-style network on MNIST; VGG16 and ResNet34 on CIFAR-10 |

Each work includes the assignment and a Russian-language report in `report/`.
Shared activation functions, losses and CSV parsing live in `common/`;
the perceptron implementations remain separate by homework.

## Build and checks

The development image contains GCC, CMake, Eigen, spdlog and CPU-only PyTorch.
No packages need to be installed on the host.

```sh
docker build -f Containerfile.dev -t iu9-neural-networks-dev .
docker run --rm --network=none -v "$PWD:/workspace:ro" iu9-neural-networks-dev
docker run --rm --network=none -v "$PWD:/workspace:ro" \
  -e SANITIZE=ON -e BUILD_TYPE=Debug iu9-neural-networks-dev
```

These commands compile all C++ programs, run numerical regression tests and
exercise PyTorch on synthetic data. They neither download datasets nor perform
full MNIST/CIFAR-10 training. Builds use a temporary directory; the checkout is
mounted read-only. CI runs the same checks.

Podman can replace Docker. On SELinux hosts, add
`--security-opt label=disable` to `run`; for writable mounts with your UID,
also use `--userns=keep-id`.

## C++ programs

Open a shell in the image, then build from the repository root:

```sh
docker run --rm -it -v "$PWD:/workspace:ro" iu9-neural-networks-dev bash
cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/build --parallel 2
/tmp/build/hw1/src/hw1
/tmp/build/hw3/src/hw3
```

`hw1` uses embedded symbols. `hw3` prints the points reached within an iteration
budget, not a guarantee of convergence; the reference minimum is `f(1, 1) = 50`.

For `hw2` and `hw4`, place MNIST CSV files in `datasets/MNIST_CSV/` before
starting the container. Each row must contain an integer label (0–9), followed
by 784 integer pixels (0–255), **without a header**. The last 10,000 rows of the
training file form the validation set; the usual 60,000-row file leaves 50,000
training examples. Pixels are scaled to [0, 1].

```sh
/tmp/build/hw2/src/hw2 datasets/MNIST_CSV/train.csv datasets/MNIST_CSV/test.csv
/tmp/build/hw4/src/hw4 datasets/MNIST_CSV/train.csv datasets/MNIST_CSV/test.csv train
/tmp/build/hw4/src/hw4 datasets/MNIST_CSV/train.csv datasets/MNIST_CSV/test.csv adam
```

`hw2` runs three loss-function experiments. In `hw4`, `train` runs a single SGD
experiment; `sgd`, `nag`, `adagrad` and `adam` select a genetic search scored on
the validation set, not the test set; numerically divergent candidates receive
zero fitness. The search is expensive: its population,
generation and epoch limits are set in `hw4/src/main.cc`.

Graphs are disabled by default. To enable the original interactive plots,
install Matplot++ and its plotting backend in your build environment and configure
with `-DNN_ENABLE_PLOTS=ON`; they are not included in the development image.

## PyTorch experiments

Datasets are downloaded only when `--download` is passed. A writable dataset
mount keeps downloads between container runs:

```sh
mkdir -p datasets
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v "$PWD/datasets:/datasets" \
  iu9-neural-networks-dev python hw5/sample/hw5.py \
  --model lenet5 --optimizer adam --epochs 20 --data-dir /datasets --download
```

Models: `lenet5`, `vgg16`, `resnet34`. Optimizers: `sgd`, `adadelta`, `nag`,
`adam`, or `all`. Each optimizer starts with a fresh model, the same seed and
the same data-order RNG state. Evaluation uses `eval()` and inference mode.
Use `--help` for batch size, learning rate and seed options. CUDA requires a
separate [CUDA-enabled PyTorch installation](https://pytorch.org/get-started/locally/);
the supplied image and automated checks use CPU only.

## Historical results

The reports and [original notebook](hw5/report/original-experiments.ipynb) retain
their original results. They predate fixes to loss evaluation, optimizers,
hyperparameter selection and PyTorch evaluation/comparison. Full training has not
been rerun, so those scores should not be treated as results of the current code.
