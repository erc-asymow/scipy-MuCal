import numpy as np
from numba import jit, vectorize, float64, int64, int32, void, intp, uintp, uint64, uint32, prange

@vectorize([float64(float64, float64)], target="parallel")
def nbadd(x, y):
  return x+y

#@jit(void(float64[:], intp[:], float64[:]), nopython=True, nogil=True, parallel=True)
@jit(nopython=True, nogil=True, parallel=True)
def add_in_place(arr, idxs, updates):
  sortidxs = np.argsort(idxs)
  idxs = idxs[sortidxs]
  updates = updates[sortidxs]
  updates = updates.astype(np.float64)
  
  mask = np.empty(idxs.shape, dtype=np.bool_)
  mask[:1] = True
  mask[1:] = idxs[1:] != idxs[:-1]
  uniqueidxs = idxs[mask]
  binidxs = mask.astype(np.intp).cumsum()

  countidxs = np.concatenate( (np.nonzero(mask)[0], np.array((mask.size,),dtype=np.intp) ))
  counts = countidxs[1:] - countidxs[:-1]
  startidxs = counts.cumsum() - counts
    
  for i in prange(uniqueidxs.shape[0]):
    startidx = startidxs[i]
    count = counts[i]
    uniquesum = np.sum(updates[startidx:startidx+count])
    arr[uniqueidxs[i]] += uniquesum


#add_in_place.parallel_diagnostics(level=4)


a = np.zeros((1024*1024*1024,), dtype=np.float64)

idxs = np.random.randint(0,a.shape[0],size=1024*1024,dtype=np.intp)
updates = np.random.rand(1024*1024).astype(np.float64)

#idxs = np.array([5,7,82,5,7,1,3,4,81,72,82], dtype=np.intp)

#updates = np.linspace(5.,50.,num=idxs.shape[0]).astype(np.float64)

#print(idxs)
#print(updates)

print("doing update")
#nbadd.at(a,idxs,updates)
add_in_place(a,idxs,updates)

print(a)


assert(0)

#print(nbadd)

#@jit(float64(float64, float64), nopython=True, nogil=True, parallel=True)
#def nbaddf(a, b):
  #return nbadd(a, b)

#assert(0)

#nbaddat = jit(nbadd.at, signature = void(float64, intp, float64), nopython=True)

#@jit(void(float64, intp, float64), nopython=True, nogil=True, parallel=False)
##@jit(nopython=True, nogil=True, parallel=True)
#def add_in_place(arr, idxs, update):
##def add_in_place(arr):
  ##np.add.at(arr,idxs,B=update)
  ##idxs = np.array([0,1,2,3]).astype(np.intp)
  ##update = np.ones_like(idxs).astype(np.float64)
  #return nbadd.at(arr,idxs,B=update)
  ##nbadd(arr[idxs], update)
  ##np.add(arr[idxs], update)
  ##arr[idxs] += update
  
##add_in_place.inspect_types()
##assert(0)
  
a = np.zeros((100,),dtype=np.float64)
a2 = np.zeros((100,),dtype=np.float64)
a3 = np.zeros((100,),dtype=np.float64)

idxs = np.array([5,7,82,5,7,1,3,4,81,72,82], dtype=np.intp)

update = np.linspace(5.,50.,num=idxs.shape[0]).astype(np.float64)

print(a)
print(idxs)
print(update)

print(a.dtype)
print(idxs.dtype)
print(update.dtype)

#add_in_place(a,idxs,update)
#add_in_place(a)
#nbadd.at(a,idxs,update)
nbadd(a[idxs], update)
np.add.at(a2,idxs,update)
a3[idxs] += update

#add_in_place.parallel_diagnostics(level=4)


print(a)
print(a2)
print(a3)
