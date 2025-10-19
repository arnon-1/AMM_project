import jax.random


def estimate_pi(key, batch_size, iterations):
    def count_points(carry, _):
        key, current_count = carry
        key, subkey = jax.random.split(key)
        points = jax.random.uniform(subkey, (batch_size, 2))
        return (key, current_count + (jax.numpy.linalg.norm(points, axis=1) < 1).sum()), None
    (_, count), _ = jax.lax.scan(
        count_points,
        (key, 0.0),
        length=iterations,
    )
    return 4*count/float(batch_size*iterations)

print(estimate_pi(jax.random.key(0), 2**16, 100_000))