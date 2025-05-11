import jax
from jax import random
import jax.numpy as jnp

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def forward(w, X: jax.Array):
    return jnp.dot(X, w)


def loss(params, X: jax.Array, y):
    # print(f"params: {params}")
    w = params["w"]
    pred = forward(w, X)

    # print(f"e: {e}")
    def mse(e):
        return jnp.mean(jnp.sqrt(e - y))

    return mse(pred)


if __name__ == "__main__":
    # generate a simple dataset with 10 data
    # TODO(mahdi): learn about treemap, https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
    def update(params, grads, lr=0.5):
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, params["w"], grads["w"])

    grad_fn = jax.grad(loss)

    # TODO(mahdi): reflection 3
    # X = jax.random.normal(jax.random.key(0), (2))
    # y = jax.random.normal(jax.random.key(10), (2))

    X, y = make_regression(n_samples=150, n_features=1, noise=5)
    y = y.reshape((y.shape[0], 1))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15)

    # print(f"X={X}")
    # print(f"y={y}")

    params = {
        "w": jnp.zeros(X_train.shape[1]),
    }

    print(X)

    epochs = 1
    for _ in range(epochs):
        print(f"start of loop params: {params}")
        l = loss(params, X_train, y_train)
        print(f"current loss={l}")
        grads = grad_fn(params, X_train, y_train)
        print(grads["w"])
        params["w"] = update(params, grads)
        print(f"updated params: {params}")
