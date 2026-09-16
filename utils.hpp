#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <deque>
#include <algorithm>
#include <random>
#include <iostream>
#include <thread>
#include <functional>

#define RELEASE_SAFELY(__POINTER) { if (__POINTER != nullptr) { delete __POINTER; __POINTER = nullptr; } }


template <typename T>
class InfiniteRandomIterator
{
    using VecType = std::vector<T>;
public:

    InfiniteRandomIterator(const VecType &v) : v(v), engine(42) {
        shuffleV();
    }

    void shuffleV(){
        std::shuffle(std::begin(v), std::end(v), engine);
        i = 0;
    }

    T next(){
        T ret = v[i++];
        if (i >= v.size()){
            if (upcoming.empty()){
                shuffleV();
            }else{
                v = std::move(upcoming.front());
                upcoming.pop_front();
                i = 0;
            }
        }
        return ret;
    }

    // Element that next() will return k calls from now (k = 0 is the next one).
    // Looking ahead does not change the sequence
    T peek(size_t k){
        size_t idx = i + k;
        size_t available = v.size();
        for (const VecType &p : upcoming) available += p.size();
        while (idx >= available){
            VecType p = upcoming.empty() ? v : upcoming.back();
            std::shuffle(std::begin(p), std::end(p), engine);
            available += p.size();
            upcoming.push_back(std::move(p));
        }
        if (idx < v.size()) return v[idx];
        idx -= v.size();
        for (const VecType &p : upcoming){
            if (idx < p.size()) return p[idx];
            idx -= p.size();
        }
        return v[0];
    }
private:
    VecType v;
    std::deque<VecType> upcoming; // permutations generated ahead by peek()
    size_t i;
    std::default_random_engine engine;
};

template <typename IndexType, typename FuncType>
void parallel_for(IndexType begin, IndexType end, FuncType func) {
    size_t range = end - begin;
    if (range <= 0) return;
    size_t numThreads = (std::min)(static_cast<size_t>(std::thread::hardware_concurrency()), range);
    size_t chunkSize = (range + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < numThreads; i++) {
        IndexType chunkBegin = begin + i * chunkSize;
        IndexType chunkEnd = (std::min)(chunkBegin + chunkSize, end);

        threads.emplace_back([chunkBegin, chunkEnd, &func]() {
            for (IndexType item = chunkBegin; item < chunkEnd; item++) {
                func(*item);
            }
        });
    }

    for (std::thread& t : threads) {
        t.join();
    }
}

#endif