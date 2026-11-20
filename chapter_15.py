import matplotlib.pyplot as plt
import numpy as np

# Truncated Fourier series for rectangle and a line (figure 15.2)
def figure_15_2():
    freqs = np.arange(0., 0.5, step=0.001)
    rads = 2 * np.pi * freqs
    f, axs = plt.subplots(2, 2, sharex=True)

    def _sinc(n, k):
        w = k * np.pi / n
        return 1. if k == 0 else np.sin(w) / w

    def get_S_n(a_coeff_fn, b_coeff_fn, n=0, lanczos:bool=False):
        '''computes the truncated fourier series from a given function that generates the coefficients'''
        k = 0
        if n > 0 and n % 2 == 0:
            k = int(n / 2) - 1
        elif n % 2 == 1:
            k = int((n - 1) / 2)

        def _s_n(w):
            weights = [1.] * (k + 1)
            if lanczos:
                weights = [_sinc(n=k, k=i) for i in range(0, k + 1)]
            return a_coeff_fn(0) + \
        sum(weights[m] * (a_coeff_fn(k=m) * np.cos(m * w) + b_coeff_fn(k=m) * np.sin(m * w)) for m in range(1, k + 1))

        def S_n(rads: np.ndarray):
            rads_ = rads.tolist()
            return np.array([_s_n(rad) for rad in rads_])
        return S_n

    def get_rect_a_k(k=0):
        '''computes the a_k coefficients for rect filter'''
        a_k = 1/2
        if k > 0 and k % 2 == 0:
            a_k = 0.
        elif k % 2 == 1:
            p = int((k - 1) / 2)
            a_k = pow(-1, p) / ((2 * p + 1) * np.pi)
        return a_k

    def get_rect_b_k(k=0):
        '''bk are zeros'''
        return 0.

    def get_line_a_k(k=0):
        '''ak are zeros'''
        return 0.

    def get_line_b_k(k=0):
        '''computes the b_k coefficients for line filter'''
        return 2 * pow(-1, k + 1) / k

    ### Upper subplot
    fn = get_S_n(get_rect_a_k, get_rect_b_k, n=3)
    curve = fn(rads - np.pi / 2)
    axs[0, 0].plot(rads, curve, label='S3')
    fn = get_S_n(get_rect_a_k, get_rect_b_k, n=11)
    curve = fn(rads - np.pi / 2)
    axs[0, 0].plot(rads, curve, label='S11')
    fn = get_S_n(get_rect_a_k, get_rect_b_k, n=19)
    curve = fn(rads - np.pi / 2)
    axs[0, 0].plot(rads, curve, label='S19')
    axs[0, 0].set_title('Rect filter Fourier expansion')
    axs[0, 0].legend()

    ### Lower subplot
    fn = get_S_n(get_line_a_k, get_line_b_k, n=3)
    curve = fn(rads)
    axs[1, 0].plot(rads, curve, label='S3')
    fn = get_S_n(get_line_a_k, get_line_b_k, n=11)
    curve = fn(rads)
    axs[1, 0].plot(rads, curve, label='S11')
    fn = get_S_n(get_line_a_k, get_line_b_k, n=19)
    curve = fn(rads)
    axs[1, 0].plot(rads, curve, label='S19')
    axs[1, 0].plot(rads, rads, linestyle='dashed', label='filter', color='black')
    axs[1, 0].set_title('Line filter fourier expansion')
    axs[1, 0].legend()

    ### same with Lanczos kernel
    fn = get_S_n(get_rect_a_k, get_rect_b_k, n=3, lanczos=True)
    curve = fn(rads - np.pi / 2)
    axs[0, 1].plot(rads, curve, label='S3')
    fn = get_S_n(get_rect_a_k, get_rect_b_k, n=11, lanczos=True)
    curve = fn(rads - np.pi / 2)
    axs[0, 1].plot(rads, curve, label='S11')
    fn = get_S_n(get_rect_a_k, get_rect_b_k, n=19, lanczos=True)
    curve = fn(rads - np.pi / 2)
    axs[0, 1].plot(rads, curve, label='S19')
    axs[0, 1].set_title('Rect filter Lanczos kernel')
    axs[0, 1].legend()

    fn = get_S_n(get_line_a_k, get_line_b_k, n=3, lanczos=True)
    curve = fn(rads)
    axs[1, 1].plot(rads, curve, label='S3')
    fn = get_S_n(get_line_a_k, get_line_b_k, n=11, lanczos=True)
    curve = fn(rads)
    axs[1, 1].plot(rads, curve, label='S11')
    fn = get_S_n(get_line_a_k, get_line_b_k, n=19, lanczos=True)
    curve = fn(rads)
    axs[1, 1].plot(rads, curve, label='S19')
    axs[1, 1].plot(rads, rads, linestyle='dashed', label='filter', color='black')
    axs[1, 1].set_title('Line filter Lanczos kernel')
    axs[1, 1].legend()

    plt.show()

if __name__ == '__main__':
    figure_15_2()
