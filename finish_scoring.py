import sys

sys.path.insert(0, '/home/ubuntu/govml')
from src.catalog import score_all

score_all(limit=None, resume=True)
print('scoring complete')
