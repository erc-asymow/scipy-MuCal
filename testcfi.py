from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("void pi();")

ffibuilder.set_source("_pi",  # name of the output C extension
"""
    void pi() { return; }
""",
    sources=[],   # includes pi.c as additional sources
    )
    

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
