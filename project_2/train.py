import sys
import os

sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])

from project_2.src.api import run_training

if __name__ == '__main__':
    run_training()
