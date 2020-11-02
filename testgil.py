import numpy as np

import concurrent.futures

def spam():
    i=0
    while True:
        print(f"spam_{i}")
        i += 1



with concurrent.futures.ThreadPoolExecutor(max_workers=32) as e:
    e.submit(spam)
    
    
    
    
