import matplotlib.pyplot as plt
import numpy as np

# chapter about digital filters, here's an illustration of nyquist sampling frequency
def nyquist_sampling():
    #### First plot 1. clean sampling: Fs/2 >> Fmax
    ## Continuous signal
    F_0 = 1_000 # hz
    T_0 = 1/F_0
    t = np.arange(0, T_0, 1/100_000)
    yt = np.sin(2 * np.pi * F_0 * t)

    ## Sampled signal
    F_s = 16_000
    t_s = np.arange(0, T_0, 1/F_s)
    y_s = np.sin(2 * np.pi * F_0 * t_s)

    plt.figure()
    plt.plot(t, yt)
    plt.xlabel('time (ms)')
    plt.hlines(0, linestyles='dashed', colors='black', xmin=0, xmax=T_0)
    plt.scatter(t_s, y_s)
    plt.vlines(t_s, ymin=0, ymax=y_s, colors='blue', linestyles='solid')
    plt.xticks(0.1 * T_0 * np.arange(0, 11, 2), range(0, 11, 2))
    plt.show()
    #### 2nd plot sampling different signals with same sample freq
    ## it can be seen from the graph that:
    # - every period of the 6K signal gets 3 samples
    # - every period of the 10K signal gets 2 samples
    F_1 = 10_000
    F_2 = 6_000
    F_s = 16_000
    T_0 = 0.001 # 1ms
    t = np.arange(0, T_0, 1/1_000_000)
    y_1 = np.sin(2 * np.pi * F_1 * t)
    y_2 = np.sin(2 * np.pi * F_2 * t)

    ts = np.arange(0, T_0, 1/F_s)
    ys_1 = np.sin(2 * np.pi * F_1 * ts)
    ys_2 = np.sin(2 * np.pi * F_2 * ts)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, y_1)
    ax1.scatter(ts, ys_1)
    ax1.hlines(y=0, colors='black', xmin=0, xmax=T_0, linestyles='-')
    ax1.vlines(ts, ymin=0, ymax=ys_1, colors='blue', linestyles='solid')
    ax1.set_title(f'F1 = {F_1:,}hz sampled at {F_s:,}hz')

    ax2.plot(t, y_2)
    ax2.scatter(ts, ys_2)
    ax2.hlines(y=0, colors='black', xmin=0, xmax=T_0, linestyles='-')
    ax2.vlines(ts, ymin=0, ymax=ys_2, colors='blue', linestyles='solid')
    ax2.set_title(f'F2 = {F_2:,}hz sampled at {F_s:,}hz')
    ax2.set_xticks(0.1 * T_0 * np.arange(0, 11, 2), range(0, 11, 2))
    ax2.set_xlabel('time (ms)')
    plt.show()

def example_1_sampling_applied_ode():
    # ODE solver for y'' + y = 0; y(0) = 1; y'(0) = 0
    # solutions is y(t) = cos(2pi*t) + sin(2pi*t)
    import pandas as pd
    y_0 = 1
    y_prime_0 = 1
    n_iter = 10
    h_s = np.arange(0.1, 1, 0.3)

    def y(t, n_iter=n_iter):
        ts = np.array([k * t for k in range(n_iter)])
        return np.sin(ts) + np.cos(ts)

    def ode_solver(h, n_iter=10):
        # naive ODE solver using euler's method
        y_hat = [-1] * n_iter
        y_prime_hat = [-1] * n_iter
        for n in range(n_iter):
            if n == 0:
                y_hat[n] = y_0
                y_prime_hat[n] = y_prime_0
                continue
            y_second = - y_hat[n - 1]
            y_prime_hat[n] = h * y_second + y_prime_hat[n - 1]
            y_hat[n] = h * y_prime_hat[n] + y_hat[n - 1]
        return y_hat, y_prime_hat

    data = []
    gt = []
    for h in h_s:
        y_hat, _ = ode_solver(h, n_iter=n_iter)
        data.append(y_hat)
        gt.append(y(h, n_iter=n_iter))

    df = pd.DataFrame(
        data,
        index=[f'h = {h:.2f}' for h in h_s],
        columns=[f'iter {i}' for i in range(1, n_iter + 1)]
    ).T
    print(df)

    gt = pd.DataFrame(
        gt,
        index=[f'h = {h:.2f}' for h in h_s],
        columns=[f'iter {i}' for i in range(1, n_iter + 1)]
    ).T
    print(gt)

def rss_transfer_fns():
    def get_H_f(f, k=3):
        return np.where(
            f == 0,
            1.0,
            np.sin(k * np.pi * f) / (k * np.sin(np.pi * f))
        )

    f = plt.figure()
    freqs = np.arange(0., 0.5, step=0.01)

    for k in [3, 5]:
        plt.plot(freqs, get_H_f(freqs, k))

    plt.xlabel('Frequency f')
    plt.ylabel('H(f)')
    plt.grid(visible=True)
    plt.show()