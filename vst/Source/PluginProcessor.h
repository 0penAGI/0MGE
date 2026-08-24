#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <atomic>
#include "GranularSynth.h"
#include "CircularBuffer.h"

class ZeroGrainProcessor : public juce::AudioProcessor {
public:
    ZeroGrainProcessor();
    ~ZeroGrainProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "0MGE"; }

    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;
    GranularSynth& getSynth() { return synth; }

    std::atomic<float> audioLevel{0.0f};
    std::atomic<float> audioPeak{0.0f};
    std::atomic<float> grainLevel{0.0f};

private:
    GranularSynth synth;
    std::unique_ptr<CircularBuffer> circBufL;
    std::unique_ptr<CircularBuffer> circBufR;
    double currentSampleRate = 44100.0;
    float peakHold = 0.0f;

    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ZeroGrainProcessor)
};
