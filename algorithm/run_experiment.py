import os
import sys
from multiprocessing import Process

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(HOME + '/algorithm')
sys.path.append(HOME + '/explib')

from pathlib import Path
from stack_train import run_train, ExpConfig
from stack_predict import run_predict


def train(dataset):
    run_train(dataset)

def predict(model_dir):
    run_predict(model_dir)

if __name__ == '__main__':
    REPEATED_NUMS = 1
    datasets = [
               'omi-1',
                'omi-2'
               ]
    lst_dataset = []
    for dataset in datasets:
        lst_dataset += [dataset] * REPEATED_NUMS
    for dataset in lst_dataset:
        p = Process(target=train, args=(dataset,))
        p.start()
        p.join()

        latest_model_dir = str(sorted(Path('results').iterdir(), key=os.path.getctime)[-1])
        p = Process(target=predict, args=(latest_model_dir, ))
        p.start()
        p.join()