import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from deep_isochron.model.invertible import CouplingFlow, CubicBSpline
from scipy.interpolate import BSpline


K = 10


def rand(seed, scale=2.0, **kw):
    a, b = jax.random.split(jax.random.PRNGKey(seed))
    return CubicBSpline(
        scale * jax.random.normal(a, (K + 4,)),
        scale * jax.random.normal(b, (K - 2,)),
        **kw,
    )


m = rand(0, xy_range=(-2.0, 3.0))
xs = jnp.linspace(-1.999, 2.999, 2001)
# 1 oracle
t = np.asarray(m.knots)
a = np.asarray(m.coeffs)
tt = np.concatenate(
    [[t[0] - 1.0], t, [t[-1] + 1.0]]
)  # pad to scipy's n+k+1 knots; unused on [0,1]
ref = BSpline(tt, a, 3)
zn = (np.asarray(xs) + 2) / 5
print("1 vs scipy:", np.abs(np.asarray(jax.vmap(m)(xs)) - (-2 + 5 * ref(zn))).max())
# 2 roundtrip
worst = 0
worstg = 0
for s in range(50):
    r = rand(s, scale=3.0, xy_range=(-2.0, 3.0))
    worst = max(
        worst, float(jnp.abs(jax.jit(jax.vmap(r.inverse))(jax.vmap(r)(xs)) - xs).max())
    )
print("2 roundtrip worst over 50 splines (scale 3):", worst)
# 3 identity
z = CubicBSpline.identity(K, xy_range=(-2.0, 3.0))
print(
    "3 identity:",
    float(jnp.abs(jax.vmap(z)(xs) - xs).max()),
    float(jnp.abs(jax.vmap(z.inverse)(xs) - xs).max()),
)
# 4 C2 at joins and knots
d1 = jax.grad(m)
d2 = jax.grad(d1)
d3 = jax.grad(d2)
for e in (-2.0, 3.0):
    print(
        "4 join",
        e,
        "f-x:",
        float(m(e + 0.0) - e),
        " f' in/out:",
        float(d1(e - 1e-9 * np.sign(e))),
        float(d1(e + 1e-9 * np.sign(e))),
        " f'' in:",
        float(d2(e - 1e-9 * np.sign(e))),
    )
kn = -2 + 5 * t[3 : K + 2]
print(
    "  interior knots: max jump f':",
    max(abs(float(d1(k + 1e-10) - d1(k - 1e-10))) for k in kn),
    " f'':",
    max(abs(float(d2(k + 1e-10) - d2(k - 1e-10))) for k in kn),
    " f''' (expected O(1)):",
    max(abs(float(d3(k + 1e-10) - d3(k - 1e-10))) for k in kn),
)
print("  min f' on range:", float(jax.vmap(d1)(xs).min()))
# 5 inverse derivatives via implicit rule
ys = jax.vmap(m)(xs[::50])
gi = jax.vmap(jax.grad(m.inverse))(ys)
print("5 (f^-1)' * f' - 1:", float(jnp.abs(gi * jax.vmap(d1)(xs[::50]) - 1).max()))
g2 = jax.vmap(jax.grad(jax.grad(m.inverse)))(ys)
fx = jax.vmap(d1)(xs[::50])
fxx = jax.vmap(d2)(xs[::50])
print("  (f^-1)'' vs -f''/f'^3:", float(jnp.abs(g2 + fxx / fx**3).max()))
# param grads of inverse vs finite differences, and finiteness incl. out-of-range
pts = jnp.array([-50.0, -2.0, -1.3, 0.2, 2.7, 3.0, 40.0])
loss = lambda mm: jnp.sum(jax.vmap(mm.inverse)(pts) ** 2)
g = eqx.filter_grad(loss)(m)
h = 1e-6
e0 = jnp.zeros(K + 4).at[5].set(h)
fd = (
    loss(eqx.tree_at(lambda q: q._dt, m, m._dt + e0))
    - loss(eqx.tree_at(lambda q: q._dt, m, m._dt - e0))
) / (2 * h)
print(
    "  param grad vs FD:",
    float(g._dt[5]),
    float(fd),
    " finite:",
    bool(jnp.isfinite(g._dt).all() & jnp.isfinite(g._dalpha).all()),
)
rg = jax.jit(jax.jacrev(lambda mm: jax.vmap(mm.inverse)(pts)))(m)
print("  reverse-mode OK:", bool(jnp.isfinite(rg._dt).all()))


keys = jax.random.split(jax.random.PRNGKey(0), 4)
layers = [
    CouplingFlow(
        2,
        CubicBSpline.identity(16, xy_range=(-3.0, 3.0)),
        flip=bool(i % 2),
        mlp_width=32,
        key=k,
    )
    for i, k in enumerate(keys)
]
X = jax.random.uniform(
    keys[0], (256, 2), minval=-4, maxval=4
)  # some points outside the box


def fwd(ls, x):
    for l in ls:
        x = l(x)
    return x


def inv(ls, y):
    for l in reversed(ls):
        y = l.inverse(y)
    return y


print("num_params per spline:", layers[0].bijection_factory.num_params, "(= 2K+2)")
print(
    "identity at init:", float(jnp.abs(jax.vmap(lambda x: fwd(layers, x))(X) - X).max())
)


# randomise final layers, then check round trip + grads + Hessian finiteness
def perturb(l, k):
    W = l.mlp.layers[-1]
    return eqx.tree_at(
        lambda m: (m.mlp.layers[-1].weight, m.mlp.layers[-1].bias),
        l,
        (jax.random.normal(k, W.weight.shape), jax.random.normal(k, W.bias.shape)),
    )


layers = [perturb(l, k) for l, k in zip(layers, keys)]
Y = jax.jit(jax.vmap(lambda x: fwd(layers, x)))(X)
print(
    "flow roundtrip:",
    float(jnp.abs(jax.jit(jax.vmap(lambda y: inv(layers, y)))(Y) - X).max()),
)
loss = lambda ls: (
    jnp.mean(jax.vmap(lambda y: inv(ls, y))(Y) ** 2)
    + jnp.mean(jax.vmap(lambda x: fwd(ls, x))(X) ** 2)
)
g = eqx.filter_jit(eqx.filter_grad(loss))(layers)
print(
    "grads finite:",
    all(bool(jnp.isfinite(l).all()) for l in jax.tree_util.tree_leaves(g)),
)
H = jax.jit(jax.vmap(jax.hessian(lambda x: fwd(layers, x))))(X)
Hi = jax.jit(jax.vmap(jax.hessian(lambda y: inv(layers, y))))(Y)
print(
    "Hessians of H and H^-1 finite:",
    bool(jnp.isfinite(H).all()),
    bool(jnp.isfinite(Hi).all()),
    " max|D2H|:",
    float(jnp.abs(H).max()),
)
