import ROOT
import numpy as np

ROOT.gInterpreter.Declare("""

#include <atomic>
#include <vector>

std::vector<double> test() {
  return std::vector<double>(60000ull*60000ull);
}

std::vector<double> test2() {
  const unsigned long long size = 60000ull*60000u;
  //std::atomic<double>* dataatom = new std::atomic<double>[size];
  std::unique_ptr<std::atomic<double>[]> dataatom = std::unique_ptr<std::atomic<double>[]>(new std::atomic<double>[size]);
  double* data = reinterpret_cast<double*>(dataatom.get());
  return std::vector<double>(std::move(data), std::move(data+size));
}

std::vector<std::atomic<double> > testatom() {
  return std::vector<std::atomic<double> >(60000ull*60000ull);
}

std::vector<double> testatomswap() {
  std::vector<std::atomic<double> > vecatom(60000ull*60000ull);
  std::vector<double> veccast = *reinterpret_cast<std::vector<double>*>(&vecatom);
  std::vector<double> vec;
  vec.swap(veccast);
  return vec;
}




""")

#ROOT.gInterpreter.ProcessLine(".L testpyroot.c+")


vec = ROOT.test2()
#vec = ROOT.testatom()
#vec = ROOT.testatomswap()
#carr = ROOT.testptr()
#rvec = ROOT.testrvec()
#print(type(vec))

print("constructing array")
#arr = np.array(vec)

#arr = np.frombuffer(carr, dtype=np.float64)
arr = np.frombuffer(vec.data(), dtype=np.float64)

#arr = np.frombuffer(vec.data(), dtype=np.float64)
print("done array")
print(arr.shape)

#print(vec)
