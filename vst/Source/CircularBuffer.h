#pragma once
#include <atomic>
#include <cstring>
#include <vector>

class CircularBuffer {
public:
    explicit CircularBuffer(int capacity)
        : buf(capacity), writePos(0), capacity(capacity) {}

    void write(const float* data, int numSamples) {
        int wp = writePos.load(std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i) {
            buf[(wp + i) % capacity] = data[i];
        }
        writePos.store((wp + numSamples) % capacity, std::memory_order_release);
    }

    void read(float* dest, int numSamples, int offsetFromLatest) const {
        int wp = writePos.load(std::memory_order_acquire);
        int start = (wp - offsetFromLatest - numSamples + capacity * 2) % capacity;
        for (int i = 0; i < numSamples; ++i) {
            dest[i] = buf[(start + i) % capacity];
        }
    }

    float readSample(int offsetFromLatest) const {
        int wp = writePos.load(std::memory_order_acquire);
        int idx = (wp - offsetFromLatest - 1 + capacity * 2) % capacity;
        return buf[idx];
    }

    int getCapacity() const { return capacity; }

private:
    std::vector<float> buf;
    std::atomic<int> writePos;
    int capacity;
};
