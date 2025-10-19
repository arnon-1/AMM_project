import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from matplotlib_settings import plt

N_particles = 16
batch_size = 32
box_length = 50

sigma_step = 0.3 * (2/N_particles)**2    # arbitrary heuristic for stddev of proposal
n_steps = 300_000        # total MH steps
burn_in = 200_000         # burn-in steps (not plotted)
T_multiplier = 70  # = 4*ε / (k_B*T)


def vdw_potential(dist, width=1.0):
    d6 = (width/dist)**6
    return (d6**2 - d6)*T_multiplier

@jit
def all_pairs_vdw(loc):
    # loc: (N,3)
    diffs = loc[:, None, :] - loc[None, :, :]          # (N,N,3)
    d2 = jnp.sum(diffs * diffs, axis=-1)               # (N,N)

    # Put +inf ONLY on the diagonal (avoid 0*inf -> NaN)
    eye = jnp.eye(d2.shape[0], dtype=bool)
    d2 = jnp.where(eye, jnp.inf, d2)

    # Upper-tri mask (exclude diagonal)
    mask = jnp.triu(jnp.ones_like(d2, dtype=bool), k=1)

    r = jnp.sqrt(d2)  # TODO: refactor for faster plotting?
    pot = jnp.where(mask, vdw_potential(r), 0.0)
    return jnp.sum(pot)

master_key = jax.random.PRNGKey(0)
key = jax.random.key(0)
key, subkey = jax.random.split(key)
split_keys = jax.random.split(key, batch_size)
pos = jax.random.normal(subkey, (batch_size, N_particles, 3))
pos %= box_length
batched_vdw = jax.vmap(all_pairs_vdw)(pos)



def mh_chain(key, pos0, vdw0, batch_size, n_steps, sigma_step0, adaption_steps=100, target_acceptance=0.3):
    """
    Run a Metropolis–Hastings chain.

    key: PRNGKey
    pos0: (N, D) initial positions
    vdw0: scalar energy at pos0 (all_pairs_vdw(pos0))
    batch_size: not actually used, but it is actually batched
    n_steps: number of MH steps
    sigma_step0: starting proposal stddev (same shape-broadcast as pos0)
    adaption_steps: number of steps to estimate acceptance for updating stddev
    target_acceptance: will optimise to this acceptance rate
    """
    sigma_step0 = jnp.asarray(sigma_step0)
    if n_steps%adaption_steps:
        raise ValueError("adaption steps not a divisor of total steps")

    def adaptive_sigma_step(carry, _):
        key, pos, vdw, sigma_step = carry
        def one_step(carry, _):
            key, pos, vdw = carry
            key, k_prop, k_u = jax.random.split(key, 3)

            proposal = pos + jax.random.normal(k_prop, pos.shape) * sigma_step
            proposal = proposal % box_length
            new_vdw = all_pairs_vdw(proposal)
            dE = new_vdw - vdw

            # Compare in log-space for numerical stability:
            log_u = jnp.log(jax.random.uniform(k_u) + 1e-12)
            accept = log_u < jnp.minimum(0.0, -dE)

            pos_next = jnp.where(accept, proposal, pos)
            vdw_next = jnp.where(accept, new_vdw, vdw)

            return (key, pos_next, vdw_next), (pos_next, accept)
        batch_step = jax.vmap(one_step)

        (key_f, pos_f, vdw_f), (positions, accepts) = lax.scan(
            batch_step,
            (key, pos, vdw),
            xs=None,
            length=adaption_steps,
        )
        accept_chance = accepts.sum()/accepts.size
        sigma_step = jax.lax.select(accept_chance > target_acceptance, sigma_step*1.1, sigma_step*0.9)
        return (key_f, pos_f, vdw_f, sigma_step), (positions, accepts)

    (key_f, pos_f, vdw_f, sigma_step), (positions, accepts) = lax.scan(
        adaptive_sigma_step,
        (key, pos0, vdw0, sigma_step0),
        xs=None,
        length=n_steps//adaption_steps,
    )
    positions = positions.reshape(n_steps, *positions.shape[2:]).swapaxes(0, 1)
    return positions, accepts


positions, accepts = mh_chain(
    split_keys, pos, batched_vdw, batch_size, n_steps, sigma_step
)
print(f"done running, average acceptance is {accepts.sum()/accepts.size}")

def _pairwise_distances(frame):
    """All unique pair distances for a single frame (N,3)."""
    diff = frame[:, None, :] - frame[None, :, :]       # (N,N,3)
    d = np.sqrt(np.sum(diff * diff, axis=-1))          # (N,N)
    i, j = np.triu_indices(frame.shape[0], k=1)
    return d[i, j]                                      # (N*(N-1)/2,)

def distances_histogram(traj, burn_in=0, stride=10, bins=1000, density=True):
    """
    traj: ndarray, shape (T, N, 3) or (B, T, N, 3)
    burn_in: discard first burn_in steps
    stride: keep every 'stride' steps after burn-in, lower stride gives nicer plot but takes longer
    """
    traj = np.asarray(traj)

    # Handle both (T,N,3) and (B,T,N,3)
    if traj.ndim == 3:
        traj_sel = traj[burn_in::stride]                # (T', N, 3)
    elif traj.ndim == 4:
        traj_sel = traj[:, burn_in::stride]             # (B, T', N, 3)
        traj_sel = traj_sel.reshape(-1, *traj_sel.shape[-2:])  # (B*T', N, 3)
    else:
        raise ValueError(f"Expected traj.ndim 3 or 4, got {traj.ndim}")

    all_dists = []
    for frame in traj_sel:
        all_dists.append(_pairwise_distances(frame))
    if not all_dists:
        raise ValueError("No frames selected. Check burn_in/stride vs. traj length.")

    all_dists = np.concatenate(all_dists)

    plt.figure()
    all_dists = all_dists[all_dists <= 1.4]
    plt.hist(all_dists, bins=bins, density=density)
    plt.xlabel("Distance r")
    plt.ylabel("Density" if density else "Count")
    plt.show()

    return all_dists


# positions: np.ndarray of shape (T, 64, 3)
_ = distances_histogram(positions, burn_in=burn_in, stride=20, bins=1000)
