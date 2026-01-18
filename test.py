import os, time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from verl.utils.reward_score.seccodeplt import compute_score

def worker(code_str: str):
    return compute_score(code_str)

if __name__ == "__main__":
    with open("n_code.txt", "r", encoding="utf-8") as f:
        traj = f.read()

    N = 1000
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=int(os.cpu_count()/2)) as ex:
        results = list(tqdm(ex.map(worker, [traj] * N, chunksize=8), total=N))

    end_time = time.time()
    print(f"[result] = {results[-1]} with elapsed time = {end_time - start_time:.2f}s")
    print("[Yeah]")
