from argparse import Namespace, ArgumentParser

from numpy import ndarray, random, stack
from numpy.ma import sqrt


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-num_particles', type=int, default=2, help='Number of particles to include in the system')

    options = parser.parse_args()

    return options


def probability_constrain(amplitudes: ndarray) -> bool:
    probabilities: ndarray = amplitudes**2
    if probabilities.sum(axis=1).all() == 1:
        return True
    return False


def generate_random(dimensions: int) -> ndarray:
    amp1 = random.random(dimensions) + random.random() * 1j
    amp2 = sqrt(1.0+0j - amp1**2)

    return stack([amp1, amp2], axis=1)
