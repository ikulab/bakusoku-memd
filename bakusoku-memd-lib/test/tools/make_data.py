import numpy as np


def make_signal(step: int, ch: int, seed: int):
    np.random.seed(0)
    inp = np.random.rand(step, ch)
    np.savetxt(f"../res/ndarray_{step}_{ch}_seed{seed}.csv", inp, delimiter=',')


def make_validator(step: int, ch: int, seed: int, ndim: int):
    from memd import memd
    inp = np.loadtxt(f"../res/ndarray_{step}_{ch}_seed{seed}.csv", delimiter=',')
    imf = memd(inp, ndim)
    for i in range(imf.shape[0]):
        np.savetxt(f"../res/ndarray_{step}_{ch}_seed{seed}_{ndim}dim_imf{i:03}.csv", imf[i].transpose(), delimiter=',')


if __name__ == '__main__':
    step = 173
    ch = 5
    seed = 0
    make_signal(step, ch, seed)
    # make_validator(step, ch, seed, 11)
