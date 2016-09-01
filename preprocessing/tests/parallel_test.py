from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to

# perform on each input. For example...

inputs = range(10)

def processInput(i):
    return i * i

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
# print results
