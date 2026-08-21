#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

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
        float minSizeMs = 30.0f + paramSize * 50.0f;
        float maxSizeMs = minSizeMs + 80.0f + paramSize * 220.0f;
        float baseGrainMs = (minSizeMs + maxSizeMs) * 0.5f;
        int baseGrainSamples = (int)(baseGrainMs * inputSR / 1000.0f);
        baseGrainSamples = std::max(256, std::min(baseGrainSamples, circLen / 4));

        int hopSamples = baseGrainSamples / 2;
        hopSamples = std::max(64, hopSamples);

        float rate = std::clamp(pitchRate * paramStretch, 0.25f, 4.0f);

        for (auto& v : voices) {
            if (!v.active) continue;
            renderVoice(v, outL, outR, numSamples, circLen);
        }

        while (nextSpawnSample < numSamples) {
            spawnVoices(nextSpawnSample, baseGrainSamples, rate, circLen, bufAvail);
            nextSpawnSample += hopSamples;
        }
        nextSpawnSample -= numSamples;

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
    float getCurrentCentroid() const { return 0.5f; }
    float getInputEnergy() const { return inputEnergy; }
    int getClusterCount(int) const { return 0; }

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
        int samplesLeft;
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

    float paramDensity = 1.5f;
    float paramPitch = 0.0f;
    float paramStretch = 1.0f;
    float paramReverse = 0.0f;
    float paramMix = 0.8f;
    float paramSize = 0.5f;
    float paramScatter = 0.0f;
    float paramFreeze = 0.0f;
    float paramFocus = 0.5f;

    std::mt19937 rng;

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
            std::uniform_int_distribution<int> posDist(0, maxStart);
            int startPos = posDist(rng);

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
            v->samplesLeft = readLen;
            v->active = true;
            v->reverse = revDist(rng) < paramReverse;
        }
    }

    void renderVoice(GrainVoice& v, float* outL, float* outR, int numSamples, int circLen) {
        for (int i = 0; i < numSamples && v.active; ++i) {
            float env = 0.5f - 0.5f * std::cos(6.2831853f * v.envPhase);

            float srcPos = v.reverse ? (float)v.samplesLeft - v.readPos : v.readPos;
            float historyOffset = v.readOffset + srcPos;
            int offset0 = (int)std::floor(historyOffset);
            float frac = historyOffset - (float)offset0;

            int readIdx = ((circWrite - 1 - offset0) % circLen + circLen * 4) % circLen;
            int readIdx2 = ((circWrite - 1 - offset0 - 1) % circLen + circLen * 4) % circLen;
            readIdx = (readIdx % circLen + circLen) % circLen;
            readIdx2 = (readIdx2 % circLen + circLen) % circLen;

            float s0 = circBuf[readIdx];
            float s1 = circBuf[readIdx2];
            float sample = s0 + (s1 - s0) * frac;

            float out = sample * env * v.amp;
            mixBufL[i] += out * v.panL;
            mixBufR[i] += out * v.panR;

            v.readPos += std::abs(v.rate);
            v.envPhase += v.envInc;
            v.samplesLeft--;

            if (v.envPhase >= 1.0f || v.samplesLeft <= 0)
                v.active = false;
        }
    }
};
