
import ROOT
from numba import jit, prange

@jit(nopython=True, nogil=True)
def testrvec(arr):
    s=0.
    for v in arr:
        for val in v:
            s += val
    return s
    #for i in range(len(arr)):
        #for j

ROOT.ROOT.EnableImplicitMT()


ROOT.gInterpreter.Declare("""
    
class Testbook {

public:
    Testbook() {}
    Testbook(int i) {}
    
    //void Exec(unsigned int slot, ROOT::VecOps::RVec<float> const& vec) {
    void Exec(unsigned int slot, float pt) {
        return;
    
    }
    
    void Finalize() {}

};

""")



filename = "/data/bendavid/muoncaldata/test3/testlz4large.root"
#filename = "/data/bendavid/muoncaldata/test3/testzstdlarge.root"
#filename = "/data/bendavid/muoncaldata/test3/testzliblarge.root"
treename = "tree"

d = ROOT.ROOT.RDataFrame(treename, filename)

#d = d.Range(100000)
testbook = ROOT.Testbook()

d = d.Book(templatename("float"),testbook,["trackPt"])

d.Print()

#d = d.AsNumpy(columns=["gradv", "globalidxv"])

#print(d["globalidxv"])
#print(d["gradv"])

#sval = testrvec(d["gradv"])
#print(sval)

#hhess = d.Histo1D(("hhess", "", 100,-1.,1.), "hesspackedv")

#c1 = ROOT.TCanvas()
#hhess.Draw()

#input("wait")
