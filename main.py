from argparse import Namespace
import util
from particle import QuantumParticle

arguments = util.parse_args()


def main(args: Namespace):
    print('Welcome to the Quantum System simulator.')
    particle = QuantumParticle()
    print(particle)

    particle.measure(2)


if __name__ == '__main__':
    main(arguments)
