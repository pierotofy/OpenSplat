#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

template <typename T>
class InfiniteRandomIterator
{
    using VecType = std::vector<T>;
public:

    InfiniteRandomIterator(const VecType &v) : v(v) {
        shuffleV();
    }

    void shuffleV(){
        std::shuffle(std::begin(v), std::end(v), std::default_random_engine {});
        i = 0;
    }

    T next(){
        T ret = v[i++];
        if (i >= v.size()) shuffleV();
        return ret;
    }
private:
    VecType v;
    size_t i;
};


#endif