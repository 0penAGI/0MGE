#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <array>

class GranularSynth {
public:
    GranularSynth() : rng(std::random_device{}()) {}

    void prepare(double sampleRate) {
        inputSR = (float)sampleRate;

        int circLen = std::max(1024, (int)(sampleRate * 10.0));
        circBuf.resize(circLen, 0.0f);
        circWrite = 0;
        bufferedSamples = 0;

        mixBufL.resize(4096, 0.0f);
        mixBufR.resize(4096, 0.0f);

        feedbackL.resize(circLen, 0.0f);
        feedbackR.resize(circLen, 0.0f);
        fbWrite = 0;

        for (auto& v : voices) {
            v.active = false;
            v.envPhase = 1.0f;
        }
        nextSpawnSample = 0;
        freezeLP_L = freezeLP_R = 0.0f;
        smoothStretch = 1.0f;
        smoothSize = paramSize;

        // Spectral analysis for visualization
        visWindow.resize(VIS_FFT_SIZE);
        for (int i = 0; i < VIS_FFT_SIZE; ++i)
            visWindow[i] = 0.5f - 0.5f * std::cos(6.2831853f * i / VIS_FFT_SIZE);
        visAccum.clear();
        visWritePos = 0;
        smoothCentroid = 0.5f;
        smoothBands.fill(0.0f);
    }

    void setParameters(float dens, float pitch, float stretch, float rev, float mix,
                       float size, float scatter, float freeze, float focus) {
        paramDensity = std::clamp(dens, 0.1f, 6.0f);
        paramPitch = std::clamp(pitch, -24.0f, 24.0f);
        paramStretch = std::clamp(stretch, 0.1f, 8.0f);
        paramReverse = std::clamp(rev, 0.0f, 1.0f);
        paramMix = std::clamp(mix, 0.0f, 1.0f);
        paramSize = std::clamp(size, 0.0f, 1.0f);
        paramScatter = std::clamp(scatter, 0.0f, 1.0f);
        paramFreeze = std::clamp(freeze, 0.0f, 1.0f);
        paramFocus = std::clamp(focus, 0.0f, 1.0f);
    }

    void feedInput(const float* left, const float* right, int numSamples) {
        if (circBuf.empty() || numSamples <= 0) return;

        float frameEnergy = 0.0f;
        const int circLen = (int)circBuf.size();
        for (int i = 0; i < numSamples; ++i) {
            float mono = (left[i] + right[i]) * 0.5f;
            frameEnergy += mono * mono;
            circBuf[circWrite] = mono;
            circWrite = (circWrite + 1) % circLen;
        }

        bufferedSamples = std::min(circLen, bufferedSamples + numSamples);
        inputEnergy = std::sqrt(frameEnergy / (float)numSamples);

        for (int i = 0; i < numSamples; ++i)
            visAccum.push_back((left[i] + right[i]) * 0.5f);
        if ((int)visAccum.size() > VIS_FFT_SIZE * 2)
            visAccum.erase(visAccum.begin(),
                           visAccum.begin() + (int)visAccum.size() - VIS_FFT_SIZE);
        visWritePos += numSamples;
        if (visWritePos >= 512) {
            visWritePos = 0;
            computeSpectralBands();
        }
    }

    void processBlock(float* outL, float* outR, int numSamples) {
        if ((int)mixBufL.size() < numSamples) {
            mixBufL.resize(numSamples, 0.0f);
            mixBufR.resize(numSamples, 0.0f);
        }
        std::fill(mixBufL.begin(), mixBufL.begin() + numSamples, 0.0f);
        std::fill(mixBufR.begin(), mixBufR.begin() + numSamples, 0.0f);

        int circLen = (int)circBuf.size();
        int bufAvail = bufferedSamples;
        if (bufAvail < 512) return;

        float pitchRate = std::pow(2.0f, paramPitch / 12.0f);

        smoothStretch += (paramStretch - smoothStretch) * 0.05f;
        smoothSize    += (paramSize    - smoothSize)    * 0.05f;

        float minSizeMs = 30.0f + smoothSize * 50.0f;
        float maxSizeMs = minSizeMs + 80.0f + smoothSize * 220.0f;
        float baseGrainMs = (minSizeMs + maxSizeMs) * 0.5f;
        int baseGrainSamples = (int)(baseGrainMs * inputSR / 1000.0f);
        baseGrainSamples = std::max(256, std::min(baseGrainSamples, circLen / 4));

        int hopSamples = baseGrainSamples / 4;
        hopSamples = std::max(32, hopSamples);

        float rate = std::clamp(pitchRate * smoothStretch, 0.25f, 4.0f);

        while (nextSpawnSample < numSamples) {
            spawnVoices(nextSpawnSample, baseGrainSamples, rate, circLen, bufAvail);
            nextSpawnSample += hopSamples;
        }
        nextSpawnSample -= numSamples;

        for (auto& v : voices) {
            if (!v.active) continue;
            renderVoice(v, outL, outR, numSamples, circLen);
        }

        float normFactor = 0.25f / std::max(1.0f, paramDensity);

        for (int i = 0; i < numSamples; ++i) {
            float dryL = mixBufL[i] * normFactor;
            float dryR = mixBufR[i] * normFactor;

            if (paramFreeze > 0.5f) {
                float fbAmt = paramFreeze * 0.55f;
                int fbIdx = ((fbWrite - 1 - i + (int)feedbackL.size() * 2) % (int)feedbackL.size());
                fbIdx = ((fbIdx % (int)feedbackL.size()) + (int)feedbackL.size()) % (int)feedbackL.size();

                float cutoff = 0.02f + paramFreeze * 0.06f;
                freezeLP_L += (feedbackL[fbIdx] - freezeLP_L) * cutoff;
                freezeLP_R += (feedbackR[fbIdx] - freezeLP_R) * cutoff;

                float cross = 0.12f;
                float fbL = freezeLP_L + freezeLP_R * cross;
                float fbR = freezeLP_R + freezeLP_L * cross;

                dryL += fbL * fbAmt;
                dryR += fbR * fbAmt;
                feedbackL[fbIdx] = dryL;
                feedbackR[fbIdx] = dryR;
            }

            outL[i] = dryL;
            outR[i] = dryR;
        }

        if (paramFreeze < 0.5f)
            fbWrite = (fbWrite + numSamples) % (int)feedbackL.size();
    }

    int getPoolSize() const { return (int)voices.size(); }
    float getCurrentCentroid() const { return smoothCentroid; }
    float getInputEnergy() const { return inputEnergy; }
    int getClusterCount(int c) const { return (c >= 0 && c < 8) ? (int)(smoothBands[c] * 100.0f) : 0; }
    int getCircBufSize() const { return (int)circBuf.size(); }
    int getCircBufWritePos() const { return circWrite; }
    float getCircBufSample(int offset) const {
        int idx = ((circWrite - 1 - offset) % (int)circBuf.size() + (int)circBuf.size() * 4) % (int)circBuf.size();
        return circBuf[idx];
    }

private:
    static constexpr int MAX_VOICES = 32;

    struct GrainVoice {
        float readPos;
        float readOffset;
        float rate;
        float amp;
        float panL, panR;
        float envPhase;
        float envInc;
        int grainLen;
        int blockOffset;
        int spawnCircWrite;
        bool active;
        bool reverse;
    };

    std::array<GrainVoice, MAX_VOICES> voices{};
    int nextSpawnSample = 0;

    float inputSR = 44100.0f;
    std::vector<float> circBuf;
    int circWrite = 0;
    int bufferedSamples = 0;

    std::vector<float> mixBufL, mixBufR;
    std::vector<float> feedbackL, feedbackR;
    int fbWrite = 0;

    float inputEnergy = 0.0f;
    float freezeLP_L = 0.0f;
    float freezeLP_R = 0.0f;

    // Spectral analysis for visualization
    static constexpr int VIS_FFT_SIZE = 256;
    std::vector<float> visWindow;
    std::vector<float> visAccum;
    int visWritePos = 0;
    float smoothCentroid = 0.5f;
    std::array<float, 8> smoothBands{};

    float paramDensity = 1.5f;
    float paramPitch = 0.0f;
    float paramStretch = 1.0f;
    float paramReverse = 0.0f;
    float paramMix = 0.8f;
    float paramSize = 0.5f;
    float paramScatter = 0.0f;
    float paramFreeze = 0.0f;
    float paramFocus = 0.5f;

    float smoothStretch = 1.0f;
    float smoothSize = 0.5f;

    std::mt19937 rng;

    void computeSpectralBands() {
        if ((int)visAccum.size() < VIS_FFT_SIZE) return;

        std::vector<float> fftBuf(VIS_FFT_SIZE * 2, 0.0f);
        int start = std::max(0, (int)visAccum.size() - VIS_FFT_SIZE);
        for (int i = 0; i < VIS_FFT_SIZE; ++i)
            fftBuf[i] = visAccum[start + i] * visWindow[i];

        int j = 0;
        for (int i = 0; i < VIS_FFT_SIZE - 1; ++i) {
            if (i < j) {
                std::swap(fftBuf[2 * i], fftBuf[2 * j]);
                std::swap(fftBuf[2 * i + 1], fftBuf[2 * j + 1]);
            }
            int m = VIS_FFT_SIZE >> 1;
            while (m >= 1 && j >= m) { j -= m; m >>= 1; }
            j += m;
        }
        for (int len = 2; len <= VIS_FFT_SIZE; len <<= 1) {
            float angle = -6.2831853f / len;
            float wr = std::cos(angle), wi = std::sin(angle);
            for (int i = 0; i < VIS_FFT_SIZE; i += len) {
                float curWr = 1.0f, curWi = 0.0f;
                for (int j2 = 0; j2 < len / 2; ++j2) {
                    int u = 2 * (i + j2);
                    int v = 2 * (i + j2 + len / 2);
                    float tr = curWr * fftBuf[v] - curWi * fftBuf[v + 1];
                    float ti = curWr * fftBuf[v + 1] + curWi * fftBuf[v];
                    fftBuf[v] = fftBuf[u] - tr;
                    fftBuf[v + 1] = fftBuf[u + 1] - ti;
                    fftBuf[u] += tr;
                    fftBuf[u + 1] += ti;
                    float newWr = curWr * wr - curWi * wi;
                    curWi = curWr * wi + curWi * wr;
                    curWr = newWr;
                }
            }
        }

        // 8 bands: SUB(<120), BASS(120-400), LO-M(400-1k), MID(1k-3k), HI-M(3k-6k), HIGH(6k-12k), AIR(12k-18k), ULTRA(>18k)
        // At 48kHz/256: bin = freq * 256 / 48000 = freq / 187.5
        static constexpr int bandBins[] = {1, 2, 6, 16, 32, 64, 96, 128};
        float rawBands[8] = {};
        float numSum = 0, denSum = 0;
        for (int k = 1; k < VIS_FFT_SIZE / 2; ++k) {
            float re = fftBuf[2 * k];
            float im = fftBuf[2 * k + 1];
            float mag = re * re + im * im;
            float freq = (float)k * 187.5f;
            numSum += freq * mag;
            denSum += mag;
            for (int b = 7; b >= 0; --b) {
                if (k >= bandBins[b]) { rawBands[b] += mag; break; }
            }
        }
        float total = denSum + 1e-10f;
        for (int b = 0; b < 8; ++b)
            rawBands[b] /= total;

        float rawCentroid = (denSum > 1e-10f) ? numSum / denSum : 24000.0f;
        rawCentroid = std::clamp(rawCentroid / 24000.0f, 0.0f, 1.0f);

        float att = 0.2f, rel = 0.05f;
        for (int b = 0; b < 8; ++b) {
            if (rawBands[b] > smoothBands[b]) smoothBands[b] += (rawBands[b] - smoothBands[b]) * att;
            else smoothBands[b] += (rawBands[b] - smoothBands[b]) * rel;
        }
        smoothCentroid += (rawCentroid - smoothCentroid) * 0.1f;
    }

    void spawnVoices(int sampleOffset, int grainSamples, float baseRate, int circLen, int bufAvail) {
        int voicesToSpawn = std::max(1, (int)(paramDensity * 0.8f));
        voicesToSpawn = std::min(voicesToSpawn, 4);

        std::uniform_real_distribution<float> sizeJit(0.85f, 1.15f);
        std::uniform_real_distribution<float> ampJit(0.88f, 1.0f);
        std::uniform_real_distribution<float> panBase(-0.5f, 0.5f);
        std::uniform_real_distribution<float> rateJit(-0.08f, 0.08f);
        std::uniform_real_distribution<float> revDist(0.0f, 1.0f);

        for (int g = 0; g < voicesToSpawn; ++g) {
            GrainVoice* v = nullptr;
            for (auto& voice : voices) {
                if (!voice.active) { v = &voice; break; }
            }
            if (!v) break;

            int readLen = (int)(grainSamples * sizeJit(rng));
            readLen = std::max(256, std::min(readLen, bufAvail - 4));
            if (readLen < 256) continue;

            int maxStart = std::max(0, bufAvail - readLen - 4);
            if (maxStart < 0) continue;

            int headPos = (circWrite - readLen - 4 + circLen * 4) % circLen;
            headPos = std::min(headPos, maxStart);
            int scatterRadius = (int)(maxStart * paramScatter);
            std::uniform_int_distribution<int> posDist(-scatterRadius, scatterRadius);
            int startPos = std::clamp(headPos + posDist(rng), 0, maxStart);

            float localRate = baseRate;
            if (paramScatter > 0.05f)
                localRate *= std::pow(2.0f, rateJit(rng) * paramScatter);

            float amp = ampJit(rng);

            // Spatial spread: wider pan range, frequency-based bias
            float pan = panBase(rng);
            float spread = paramFocus * 0.3f + 0.7f;
            pan *= spread;

            v->readOffset = (float)startPos;
            v->readPos = 0.0f;
            v->rate = localRate;
            v->amp = amp;
            v->panL = std::cos((pan + 1.0f) * 3.14159265f * 0.25f);
            v->panR = std::sin((pan + 1.0f) * 3.14159265f * 0.25f);
            v->envPhase = 0.0f;
            v->envInc = 1.0f / (float)readLen;
            v->grainLen = readLen;
            v->spawnCircWrite = circWrite;
            v->blockOffset = sampleOffset;
            v->active = true;
            v->reverse = revDist(rng) < paramReverse;
        }
    }

    void renderVoice(GrainVoice& v, float* outL, float* outR, int numSamples, int circLen) {
        int start = std::min(v.blockOffset, numSamples - 1);
        v.blockOffset = 0;

        auto safeRead = [&](int offset) -> float {
            int idx = v.spawnCircWrite - 1 - offset;
            idx = ((idx % circLen) + circLen) % circLen;
            return circBuf[idx];
        };

        for (int i = start; i < numSamples && v.active; ++i) {
            float env = 0.5f - 0.5f * std::cos(6.2831853f * v.envPhase);

            float srcPos = v.reverse ? (float)v.grainLen - v.readPos : v.readPos;
            float historyOffset = v.readOffset + srcPos;
            int offset0 = (int)std::floor(historyOffset);
            float frac = historyOffset - (float)offset0;

            float s0 = safeRead(offset0);
            float s1 = safeRead(offset0 + 1);
            float sample = s0 + (s1 - s0) * frac;

            float out = sample * env * v.amp;
            mixBufL[i] += out * v.panL;
            mixBufR[i] += out * v.panR;

            v.readPos += std::abs(v.rate);
            v.envPhase += v.envInc;

            if (v.envPhase >= 1.0f || v.readPos >= (float)v.grainLen)
                v.active = false;
        }
    }
};
