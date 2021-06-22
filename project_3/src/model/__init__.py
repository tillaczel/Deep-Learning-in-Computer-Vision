from .model import Generator, Discriminator


def get_networks(f=64, blocks=9):
    return Generator(f, blocks), Generator(f, blocks), Discriminator(), Discriminator()
