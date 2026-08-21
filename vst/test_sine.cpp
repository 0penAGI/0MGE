#include <cstdio>
#include <cmath>
#include <vector>
#include "Source/GranularSynth.h"

static void writeWav(const char* path, const std::vector<float>& samples, int sr) {
    FILE* f = fopen(path, "wb");
    int n = (int)samples.size();
    int dataSize = n * 2;
    int fileSize = 36 + dataSize;
    fwrite("RIFF", 1, 4, f);
    fwrite(&fileSize, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    int fmtSize = 16;
    fwrite(&fmtSize, 4, 1, f);
    short fmt = 1; fwrite(&fmt, 2, 1, f);
    short ch = 1; fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    int byteRate = sr * 2;
    fwrite(&byteRate, 4, 1, f);
    short blockAlign = 2; fwrite(&blockAlign, 2, 1, f);
    short bits = 16; fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&dataSize, 4, 1, f);
    for (int i = 0; i < n; ++i) {
        float s = std::clamp(samples[i], -1.0f, 1.0f);
        short s16 = (short)(s * 32767.0f);
        fwrite(&s16, 2, 1, f);
    }
    fclose(f);
}

int main() {
    const int SR = 48000;
    const float FREQ = 440.0f;
    const int DURATION_SEC = 3;
    const int BLOCK = 512;

    GranularSynth synth;
    synth.prepare(SR);
    synth.setParameters(1.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.0f, 0.0f, 0.0f);

    std::vector<float> output;
    int totalSamples = SR * DURATION_SEC;

    for (int pos = 0; pos < totalSamples; pos += BLOCK) {
        int n = std::min(BLOCK, totalSamples - pos);

        std::vector<float> inL(n), inR(n);
        for (int i = 0; i < n; ++i) {
            float t = (float)(pos + i) / (float)SR;
            float sig = 0.3f * std::sin(6.2831853f * FREQ * t);
            inL[i] = sig;
            inR[i] = sig;
        }

        synth.feedInput(inL.data(), inR.data(), n);

        std::vector<float> outL(n, 0.0f), outR(n, 0.0f);
        synth.processBlock(outL.data(), outR.data(), n);

        for (int i = 0; i < n; ++i)
            output.push_back(outL[i]);
    }

    writeWav("test_sine_output.wav", output, SR);

    // Analyze: check for clicks (sample-to-sample jumps)
    int clicks = 0;
    float maxJump = 0.0f;
    float rms = 0.0f;
    for (int i = 1; i < (int)output.size(); ++i) {
        float jump = std::abs(output[i] - output[i-1]);
        if (jump > maxJump) maxJump = jump;
        if (jump > 0.1f) clicks++;
        rms += output[i] * output[i];
    }
    rms = std::sqrt(rms / (float)output.size());

    printf("Output: %d samples, %d seconds\n", (int)output.size(), DURATION_SEC);
    printf("RMS: %.4f\n", rms);
    printf("Max sample-to-sample jump: %.4f\n", maxJump);
    printf("Clicks (>0.1 jump): %d\n", clicks);
    printf("Saved: test_sine_output.wav\n");

    if (clicks > 10)
        printf("WARN: many clicks detected!\n");
    else if (maxJump > 0.05f)
        printf("WARN: large jumps present\n");
    else
        printf("OK: clean output\n");

    return 0;
}
