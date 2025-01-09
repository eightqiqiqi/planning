import numpy as np
from random import shuffle

def neighbor(cell, dir, msize):
    cell = np.array(cell, dtype=int)
    dir = np.array(dir, dtype=int)
    neigh = (cell + dir) % msize  
    return neigh

def neighbors(cell, msize, wrap=True):
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)
    Ns = [cell + dirs[a] for a in range(4)]
    as_ = list(range(4))
    if wrap:
        Ns = [N % msize for N in Ns]  
    else:
        inds = [i for i, N in enumerate(Ns) if (np.min(N) >= 0 and np.max(N) < msize)]
        Ns = [Ns[i] for i in inds]
        as_ = [as_[i] for i in inds]
    return Ns, as_

def walk(maz, nxtcell, msize, visited=None, wrap=True):
    if visited is None:
        visited = []
    dir_map = {0: 1, 1: 0, 2: 3, 3: 2} 
    visited.append((nxtcell[0] * msize) + nxtcell[1])

    neighs, as_ = neighbors(nxtcell, msize, wrap=wrap)

    for nnum in np.random.permutation(len(neighs)):
        neigh, a = neighs[nnum], as_[nnum]
        ind = (neigh[0] * msize) + neigh[1]

        if ind not in visited:
            if 0 <= neigh[0] < msize and 0 <= neigh[1] < msize and 0 <= dir_map[a] < 4:
                maz[nxtcell[0], nxtcell[1], a] = 0.0
                maz[neigh[0], neigh[1], dir_map[a]] = 0.0
                maz, visited = walk(maz, neigh, msize, visited, wrap=wrap)
            else:
                print(f"Debug: nxtcell={nxtcell}, neigh={neigh}, a={a}, dir_map[a]={dir_map[a]}")
                raise IndexError(f"Invalid indices: neigh={neigh}, dir_map[a]={dir_map[a]}, a={a}")
    return maz, visited

def maze(msize, wrap=True):
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)
    dir_map = {0: 1, 1: 0, 2: 3, 3: 2} 
    maz = np.ones((msize, msize, 4), dtype=np.float32)  
    cell = np.random.randint(0, msize, size=2)  
    maz, visited = walk(maz, cell, msize, wrap=wrap)

    if wrap:
        holes = int(3 * (msize - 3))
    else:
        holes = int(4 * (msize - 3))
        maz[msize - 1, :, 0] = 0.5
        maz[0, :, 1] = 0.5
        maz[:, msize - 1, 2] = 0.5
        maz[:, 0, 3] = 0.5

    for _ in range(holes):
        walls = np.argwhere(maz == 1)
        wall = walls[np.random.choice(len(walls))]
        cell, a = wall[:2], wall[2]

        neigh = neighbor(cell, dirs[a], msize)
        maz[cell[0], cell[1], a] = 0.0
        maz[neigh[0], neigh[1], dir_map[a]] = 0.0

    maz[maz == 0.5] = 1.0  

    maz = np.reshape(np.transpose(maz, axes=[1, 0, 2]), (msize * msize, 4))

    return maz.astype(np.float32)
