import jax
import jax.numpy as jnp


def relu(x: jax.Array):
    return jnp.maximum(0, x)


if __name__ == "__main__":
    assert relu(jnp.ones(2)).shape == jnp.array([1.0, 1.0]).shape
