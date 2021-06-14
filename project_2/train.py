import sys
import os

from project_2.src.api.train import run_training

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

if __name__ == '__main__':
    run_training()
