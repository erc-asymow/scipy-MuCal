#include<atomic>
#include <iostream>

void testlock() {
  std::atomic<double> test;
  std::cout << test.is_lock_free() << std::endl;
  
  std::cout << sizeof(test) << std::endl;
}
