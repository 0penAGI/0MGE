#include "PluginProcessor.h"
#include "PluginEditor.h"

ZeroGrainProcessor::ZeroGrainProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout()) {}

ZeroGrainProcessor::~ZeroGrainProcessor() {}

juce::AudioProcessorValueTreeState::ParameterLayout ZeroGrainProcessor::createParameterLayout() {
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"density", 1}, "Density",
        juce::NormalisableRange<float>(0.1f, 6.0f, 0.01f), 1.5f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"pitch", 1}, "Pitch",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"stretch", 1}, "Stretch",
        juce::NormalisableRange<float>(0.1f, 8.0f, 0.01f), 1.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"reverse", 1}, "Reverse",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"mix", 1}, "Mix",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.8f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"size", 1}, "Size",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"scatter", 1}, "Scatter",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.2f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"freeze", 1}, "Freeze",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.0f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"focus", 1}, "Focus",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));

    return layout;
}

void ZeroGrainProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    currentSampleRate = sampleRate;
    synth.prepare(sampleRate);

    int circSize = (int)(sampleRate * 5.0f);
    circBufL = std::make_unique<CircularBuffer>(circSize);
    circBufR = std::make_unique<CircularBuffer>(circSize);
}

void ZeroGrainProcessor::releaseResources() {
    circBufL.reset();
    circBufR.reset();
}

void ZeroGrainProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;
    int numSamples = buffer.getNumSamples();

    if (!circBufL || !circBufR) return;

    float mix = *apvts.getRawParameterValue("mix");
    synth.setParameters(
        *apvts.getRawParameterValue("density"),
        *apvts.getRawParameterValue("pitch"),
        *apvts.getRawParameterValue("stretch"),
        *apvts.getRawParameterValue("reverse"),
        mix,
        *apvts.getRawParameterValue("size"),
        *apvts.getRawParameterValue("scatter"),
        *apvts.getRawParameterValue("freeze"),
        *apvts.getRawParameterValue("focus"));

    float* inL = buffer.getWritePointer(0);
    float* inR = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : inL;

    circBufL->write(inL, numSamples);
    circBufR->write(inR, numSamples);

    synth.feedInput(inL, inR, numSamples);

    std::vector<float> outL(numSamples, 0.0f);
    std::vector<float> outR(numSamples, 0.0f);
    synth.processBlock(outL.data(), outR.data(), numSamples);

    for (int i = 0; i < numSamples; ++i) {
        inL[i] = inL[i] * (1.0f - mix) + outL[i] * mix;
        inR[i] = inR[i] * (1.0f - mix) + outR[i] * mix;
    }

    float sum = 0.0f;
    float peak = 0.0f;
    float gSum = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        float absL = std::abs(inL[i]);
        float absR = std::abs(inR[i]);
        float m = std::max(absL, absR);
        sum += inL[i] * inL[i] + inR[i] * inR[i];
        gSum += outL[i] * outL[i] + outR[i] * outR[i];
        if (m > peak) peak = m;
    }
    float rms = std::sqrt(sum / (numSamples * 2) + 1e-10f);
    float gRms = std::sqrt(gSum / (numSamples * 2) + 1e-10f);
    audioLevel.store(rms, std::memory_order_relaxed);
    grainLevel.store(gRms, std::memory_order_relaxed);
    if (peak > peakHold) peakHold = peak;
    else peakHold *= 0.995f;
    audioPeak.store(peakHold, std::memory_order_relaxed);
}

void ZeroGrainProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void ZeroGrainProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessorEditor* ZeroGrainProcessor::createEditor() {
    return new ZeroGrainEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new ZeroGrainProcessor();
}
