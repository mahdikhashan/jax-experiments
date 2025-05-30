from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from jax.example_libraries.stax import (
    serial,
    Conv,
    Relu,
    MaxPool,
    Flatten,
    Dense,
    LogSoftmax
)

debug = False

batch_size = 64

def custom_transform():
    return ToTensor()

mnist_dataset_train = MNIST('.', train=True, download=True, transform=custom_transform())
trainer = DataLoader(mnist_dataset_train, batch_size=batch_size, num_workers=0, shuffle=True)

net_init, net_params = serial(
    Conv(32, (2, 2), padding="VALID"),
    Relu,
    MaxPool((2, 2)),
    Flatten,
    Dense(128),
    Relu,
    Dense(128, 10),
    LogSoftmax
)

if __name__ == "__main__":
    if debug:
        (feature, label) = next(iter(trainer))
        print(f"train features: {feature.size()}")
        print(f"train labels: {label.size()}")

        import matplotlib.pyplot as plt

        img = feature[0].squeeze()
        label = label[0]
        plt.imshow(img, cmap="gray")
        plt.show()
    else:
        from jax import random
        rng = random.key(0)
        in_shape = (1, 28, 28, 1)
        out_shape, net_params = net_init(rng, in_shape)
