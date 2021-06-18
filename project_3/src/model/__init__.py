from .model import Generator, Discriminator


def get_networks():
    return Generator(), Discriminator()
