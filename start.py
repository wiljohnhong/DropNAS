import os
import time
import multiprocessing
import moxing as mox
import numpy as np


dataset = 'cifar10'
mox.file.copy_parallel('s3://bucket-london2/DATASETS/' + dataset.upper(),'/cache/test')
os.system('rm -r *')

base_name = '4444_final_cifar10_v1'

def single_run(seed, gpu):
    n_layers1 = 14
    n_layers2 = 20
    n_epochs1 = 76

    name = base_name + '_v' + str(gpu)

    os.system('python /cache/code/search.py --name {} --dataset {} --layers {} --epochs {} --seed {} --gpus {} --drop_rate 0.00003'
        .format(name, dataset, n_layers1, n_epochs1, seed, gpu))

    with open(os.path.join('searchs', name,'genotype.txt'), 'r') as f:
        genotype = f.read()

    os.system('python /cache/code/augment.py --name {} --dataset {} --seed {} --gpus {} --genotype "{}"'.
        format(name, dataset, seed, gpu, genotype))


if __name__ == '__main__':
    
    processes = []
    for i in range(8):
        p = multiprocessing.Process(target=single_run, args=(i, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    accs = []
    res_file = 'result_' + base_name + '.txt'
    with open(res_file, 'w') as res:
        for i in range(8):
            name = base_name + '_v' + str(i)
            with open(os.path.join('searchs', name, 'genotype.txt'), 'r') as f:
                genotype = f.read()
            with open(os.path.join('augments', name, name + '.log')) as f:
                lines = f.readlines()
                acc = lines[-1][-9:-3]
                accs.append(float(acc))
            
            res.write('{}\t{}\n'.format(acc, genotype))
        
        mean = np.mean(accs)
        std = np.std(accs)
        res.write('mean: {:.2f}\tstd: {:.2f}\n'.format(mean, std))
    
    mox.file.copy_parallel(res_file, os.path.join('s3://bucket-auto2/hongweijun/archive2/', res_file))


