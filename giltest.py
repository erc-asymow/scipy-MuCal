import lz4.block
import numpy as np
import concurrent.futures

nfloats = 1024*1024*64

rawdata = np.random.random((nfloats,))

original_bytes = rawdata.tobytes()
print(len(original_bytes))

nbytes = len(original_bytes)

print(rawdata.dtype)

compressed_bytes = lz4.block.compress(original_bytes, compression=4)

print(len(compressed_bytes))


uncompressed_bytes = lz4.block.decompress(compressed_bytes)

dest_bytes = bytearray(len(original_bytes))



lz4.block.decompress_to(compressed_bytes, dest_bytes)

print(original_bytes[:10])
print(dest_bytes[:10])

dest = np.empty((nbytes,), dtype=np.uint8)
lz4.block.decompress_to(compressed_bytes, dest)


print(dest.tobytes()[:10])

#assert(0)


print(len(uncompressed_bytes))


#print(original_bytes[:10])
##print(compressed_bytes[-10:])

#print(compressed_bytes)

#dest = bytearray(nbytes)

def dowork(compressed):
    dest = np.empty((nbytes,), dtype=np.uint8)
    lz4.block.decompress_to(compressed_bytes, dest)
    #return dest.tobytes()[:10]
    #decompressed = lz4.block.decompress(compressed)
    return True


#for i in range(512):
    #decompressed = lz4.block.decompress(compressed_bytes)

#assert(0)

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as e:
    results = []
    print("submitting")
    for i in range(100):
        results.append(e.submit(dowork, compressed_bytes))
    #results = map(dowork, 512*[compressed_bytes])
    print("done submitting")
    for res in results:
        print(res.result())
    
