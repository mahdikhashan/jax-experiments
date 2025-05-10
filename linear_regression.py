import jax
from jax import random
import jax.numpy as jnp


def forward(w, X: jax.Array):
    return jnp.dot(X, w) + params['b']

def loss(params, X: jax.Array, y):
    # print(f"params: {params}")
    w = params['w']
    e = forward(w, X) - y
    # print(e)
    def mse(e):
        return jnp.mean(jnp.sqrt(e))
    return mse(e)

if __name__ == "__main__":
    # generate a simple dataset with 10 data
    # TODO(mahdi): learn about treemap, https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html
    def update(params, grads, lr=0.5):
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, params['w'], grads)
    
    grad_fn = jax.grad(loss)

    X = jax.random.normal(jax.random.key(0), (2))
    y = jax.random.normal(jax.random.key(10), (2))

    # print(f"X={X}")
    # print(f"y={y}")

    params = {
        'w': jnp.zeros(2),
        'b': 0.0
    }

    epochs = 2
    for _ in range(epochs):
        l = loss(params, X, y)
        print(f"current loss={l}")
        grads = grad_fn(params, X, y)
        print(grads['w'])
        params['w'] = update(params, grads['w'])
        print(f"updated params: {params}")
