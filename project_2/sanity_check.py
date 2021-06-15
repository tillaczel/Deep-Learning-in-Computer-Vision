import sys
import os
sys.path.append('git_repo')
sys.path.append(os.path.split(os.getcwd())[0])
from project_2.src.metrics.sanity_check import sanity_check_metrics

sanity_check_metrics()