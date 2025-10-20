import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

dimensions = 2
J = 1
magnetic_field = 4
L = 32
batch_size = 32
n_steps = 100_000
burn_in = 50_000

starting_perturb_size = 1
T_modifier = 100  # = kb*T

key = jax.random.key(0)
key, subkey = jax.random.split(key)
split_keys = jax.random.split(key, batch_size)

s0 = jax.random.bernoulli(subkey, 0.5, (batch_size, *(dimensions*[L])))



def hamiltonian(s):
    """calculate the ising hamiltonian, s is the ising system as a boolean array"""
    assert len(s.shape) == dimensions
    neighbours_potential = 0.0
    for axis in range(dimensions):
        same_neighbours = (s == jnp.roll(s, axis=axis, shift=1)).sum()*2-s.size  # boolean logic is more effient?
        neighbours_potential = neighbours_potential - J * same_neighbours

    return neighbours_potential - magnetic_field * (2*s-1).sum()


def perturb_ising(key, s, n_flips: int):
    """perturb the ising system, flip n_flips bits"""
    # random permutation of all indices (static shape n)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, np.arange(s.size).reshape(s.shape))
    return s ^ (perm < n_flips)


def mh_chain(key, s0, pot0, batch_size, n_steps, step_size0, adaption_steps=100, target_acceptance=0.5):
    """
    Run a Metropolis–Hastings chain.

    key: PRNGKey
    s0: batch of ising systems
    pot0: scalar energy at s0 (all_pairs_pot(s0))
    batch_size: not actually used, but it is actually batched
    n_steps: number of MH steps
    step_size: starting proposal stddev (same shape-broadcast as s0)
    adaption_steps: number of steps to estimate acceptance for updating stddev
    target_acceptance: will optimise to this acceptance rate
    """
    step_size0 = jnp.asarray(step_size0)
    if n_steps%adaption_steps:
        raise ValueError("adaption steps not a divisor of total steps")

    def adaptive_sigma_step(carry, _):
        key, pos, pot, step_size = carry
        def one_step(carry, _):
            key, s, pot = carry
            key, perturb_key, k_u = jax.random.split(key, 3)

            proposal = perturb_ising(perturb_key, s, step_size)
            new_pot = hamiltonian(proposal)
            dE = new_pot - pot

            # Compare in log-space for numerical stability:
            log_u = jnp.log(jax.random.uniform(k_u) + 1e-12)
            accept = log_u < jnp.minimum(0.0, -dE/T_modifier)

            pos_next = jnp.where(accept, proposal, s)
            pot_next = jnp.where(accept, new_pot, pot)

            return (key, pos_next, pot_next), (pos_next, accept)
        batch_step = jax.vmap(one_step)

        (key_f, s_f, pot_f), (ising_systems, accepts) = lax.scan(
            batch_step,
            (key, pos, pot),
            xs=None,
            length=adaption_steps,
        )
        accept_chance = accepts.sum()/accepts.size
        step_size = jax.lax.select(accept_chance > target_acceptance, step_size+1, jnp.maximum(step_size-1,1))
        return (key_f, s_f, pot_f, step_size), (ising_systems, accepts)

    (key_f, s_f, pot_f, step_size_f), (ising_systems, accepts) = lax.scan(
        adaptive_sigma_step,
        (key, s0, pot0, step_size0),
        xs=None,
        length=n_steps//adaption_steps,
    )
    ising_systems = ising_systems.reshape(n_steps, *ising_systems.shape[2:]).swapaxes(0, 1)
    return ising_systems, accepts, step_size_f

ising_systems, accepts, step_size_f = mh_chain(
    split_keys, s0, jax.vmap(hamiltonian)(s0), batch_size, n_steps, starting_perturb_size
)

print(f"done running, average acceptance is {accepts.sum()/accepts.size}")
print("final step size:", step_size_f)




from matplotlib_settings import plt

# ising_systems has shape: (batch_size, n_steps, L, L)
# Convert booleans -> spins
spins = (ising_systems.astype(jnp.int8) * 2 - 1)  # (B, T, L, L)

# Magnetisation per configuration (average spin)
# shape: (B, T)
magnetisation = spins.mean(axis=-np.arange(dimensions)-1)

stride = 10
magnetisation = magnetisation[:, burn_in::stride]  # (B, T')
magnetisation_all = np.asarray(magnetisation).ravel()

# Histogram in [-1, 1]
bins = 300
counts, bin_edges = np.histogram(magnetisation_all, bins=bins, range=(-1, 1), density=True)



plt.figure()
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.bar(centers, counts, width=centers[1]-centers[0])
plt.xlabel("Magnetisation m")
plt.ylabel("Density")
plt.show()
