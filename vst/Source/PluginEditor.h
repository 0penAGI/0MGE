#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "PluginProcessor.h"
#include <cmath>
#include <algorithm>

class CleanSliderLookAndFeel : public juce::LookAndFeel_V4 {
public:
    void drawTextEditorOutline(juce::Graphics&, int, int, juce::TextEditor&) override {}
    void drawPopupMenuBackground(juce::Graphics& g, int w, int h) override {
        g.fillAll(juce::Colour(0xfff8f8f8));
    }
    juce::Font getPopupMenuFont() override { return juce::Font(12.0f, juce::Font::plain); }
};

class ZeroGrainEditor : public juce::AudioProcessorEditor, public juce::Timer {
public:
    ZeroGrainEditor(ZeroGrainProcessor& p)
        : AudioProcessorEditor(&p), proc(p)
    {
        setSize(560, 680);
        setRepaintsOnMouseActivity(false);

        for (auto& pt : particles)
            resetParticle(pt, true);

        auto setupKnob = [&](juce::Slider& s, const juce::String& paramId,
                             const juce::String& label, int x, int y) {
            s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
            s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 16);
            s.setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xff111111));
            s.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0x00000000));
            s.setColour(juce::Slider::thumbColourId, juce::Colour(0x00000000));
            s.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xff111111));
            s.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colour(0x00000000));
            s.setColour(juce::Slider::textBoxOutlineColourId, juce::Colour(0x00000000));
            s.setLookAndFeel(&cleanLaf);
            addAndMakeVisible(s);

            auto* att = new juce::SliderParameterAttachment(
                *proc.apvts.getParameter(paramId), s, nullptr);
            attachments.add(att);

            auto* lbl = new juce::Label({}, label);
            lbl->setJustificationType(juce::Justification::centred);
            lbl->setFont(juce::Font(12.0f, juce::Font::bold));
            lbl->setColour(juce::Label::textColourId, juce::Colour(0xff333333));
            addAndMakeVisible(lbl);
            lbl->setBounds(x, y, 80, 14);
        };

        int knobY = 90;
        int knobTop = knobY + 18;
        int knobH = 82;

        setupKnob(densityKnob, "density", "DENSITY", 20, knobY);
        setupKnob(pitchKnob, "pitch", "PITCH", 105, knobY);
        setupKnob(stretchKnob, "stretch", "STRETCH", 190, knobY);
        setupKnob(reverseKnob, "reverse", "REVERSE", 275, knobY);
        setupKnob(sizeKnob, "size", "SIZE", 360, knobY);
        setupKnob(scatterKnob, "scatter", "SCATTER", 445, knobY);

        densityKnob.setBounds(20, knobTop, 80, knobH);
        pitchKnob.setBounds(105, knobTop, 80, knobH);
        stretchKnob.setBounds(190, knobTop, 80, knobH);
        reverseKnob.setBounds(275, knobTop, 80, knobH);
        sizeKnob.setBounds(360, knobTop, 80, knobH);
        scatterKnob.setBounds(445, knobTop, 80, knobH);

        int row2Y = knobTop + knobH + 28;
        setupKnob(mixKnob, "mix", "MIX", 145, row2Y);
        setupKnob(freezeKnob, "freeze", "FREEZE", 240, row2Y);
        setupKnob(focusKnob, "focus", "FOCUS", 335, row2Y);

        mixKnob.setBounds(145, row2Y + 18, 80, knobH);
        freezeKnob.setBounds(240, row2Y + 18, 80, knobH);
        focusKnob.setBounds(335, row2Y + 18, 80, knobH);

        vizY = row2Y + knobH + 24;
        vizH = 220;

        startTimerHz(30);
    }

    void paint(juce::Graphics& g) override {
        g.fillAll(juce::Colour(0xfff8f8f8));

        float audioEnergy = std::clamp(proc.audioLevel.load(std::memory_order_relaxed) * 8.0f, 0.0f, 1.0f);
        float freeze = *proc.apvts.getRawParameterValue("freeze");

        // Background pulse with audio
        if (audioEnergy > 0.05f) {
            float pulseAlpha = audioEnergy * 0.04f;
            g.setColour(juce::Colour::fromRGBA(0, 153, 255, (juce::uint8)(pulseAlpha * 255)));
            g.fillAll();
        }

        // Logo
        g.setColour(juce::Colour(0xff222222));
        g.setFont(juce::Font(36.0f, juce::Font::plain));
        g.drawText("0MGE", getWidth() / 2 - 55, 8, 110, 46, juce::Justification::centred);

        g.setColour(juce::Colour(0xffbbbbbb));
        g.setFont(juce::Font(10.0f, juce::Font::plain));
        g.drawText("granular engine", getWidth() / 2 - 55, 52, 110, 14, juce::Justification::centred);

        // Viz area background
        float freezeTint = freeze > 0.5f ? (freeze - 0.5f) * 2.0f : 0.0f;
        if (freezeTint > 0.01f) {
            // Frost overlay — icy white/blue gradient
            g.setColour(juce::Colour::fromRGBA(180, 210, 240, (juce::uint8)(freezeTint * 18)));
            g.fillRoundedRectangle(12.0f, (float)vizY - 4.0f, (float)(getWidth() - 24), (float)(vizH + 8), 8.0f);

            // Crystalline frost lines
            float fw = (float)(getWidth() - 24);
            float fh = (float)vizH;
            float fy = (float)vizY;
            juce::Random rng(42);
            g.setColour(juce::Colour::fromRGBA(200, 230, 255, (juce::uint8)(freezeTint * 40)));
            for (int i = 0; i < 6; ++i) {
                float lx = 12.0f + rng.nextFloat() * fw;
                float ly = fy + rng.nextFloat() * fh;
                float len = 15.0f + rng.nextFloat() * 25.0f;
                float angle = rng.nextFloat() * 6.28f;
                float ex = lx + std::cos(angle) * len;
                float ey = ly + std::sin(angle) * len;
                g.drawLine(lx, ly, ex, ey, 0.5f + freezeTint * 0.5f);
            }
        } else {
            g.setColour(juce::Colour::fromRGBA(0, 120, 220, 0));
            g.fillRoundedRectangle(12.0f, (float)vizY - 4.0f, (float)(getWidth() - 24), (float)(vizH + 8), 8.0f);
        }

        drawClusterBars(g);
        drawWaveform(g);
        drawParticles(g);

        // Footer
        int grains = proc.getSynth().getPoolSize();
        g.setColour(juce::Colour(0xff999999));
        g.setFont(juce::Font(9.0f, juce::Font::plain));
        g.drawText("grains: " + juce::String(grains),
                     20, getHeight() - 20, 120, 14, juce::Justification::centredLeft);

        g.setColour(juce::Colour(0xffbbbbbb));
        g.setFont(juce::Font(9.0f, juce::Font::plain));
        g.drawText("by 0penAGI",
                     getWidth() - 300, getHeight() - 20, 280, 14, juce::Justification::centredRight);
    }

    void resized() override {}

    void timerCallback() override {
        float plugW = (float)getWidth();
        float plugH = (float)getHeight();

        float vizTop = (float)vizY;
        float vizHf = (float)this->vizH;

        float freeze = *proc.apvts.getRawParameterValue("freeze");
        float freezeFactor = freeze > 0.5f ? (freeze - 0.5f) * 2.0f : 0.0f;
        float freezeDamp = 1.0f - freezeFactor * 0.3f;

        // Fetch actual grain data from engine
        GranularSynth::GrainInfo grainData[GranularSynth::MAX_GRAINS];
        int activeCount = proc.getSynth().getActiveGrains(grainData, GranularSynth::MAX_GRAINS);

        for (int i = 0; i < NUM_PARTICLES; ++i) {
            auto& pt = particles[i];
            if (i < activeCount) {
                auto& g = grainData[i];
                // Grain is alive — drive particle from grain state

                // X = pan position (left=0..1=right), spread across viz width
                float panCenter = (g.panL + g.panR); // 0..~1.4
                float panNorm = std::clamp((g.panL - g.panR + 1.0f) * 0.5f, 0.0f, 1.0f);
                float targetX = 12.0f + panNorm * (plugW - 24.0f);

                // Y = readPos01 within grain (top=start, bottom=end)
                float targetY = vizTop + g.readPos01 * vizHf;

                pt.vx += (targetX - pt.x) * 0.15f * freezeDamp;
                pt.vy += (targetY - pt.y) * 0.15f * freezeDamp;
                pt.vx *= 0.85f;
                pt.vy *= 0.85f;
                pt.x += pt.vx;
                pt.y += pt.vy;

                // Size = envelope * amplitude
                pt.currentSize = (1.5f + g.amp * 5.0f) * (0.3f + g.env * 0.7f);

                // Life = envelope (drives alpha in drawParticles)
                pt.life = g.env;

                // Phase for glow animation
                pt.phase += 0.06f * std::abs(g.rate);

                // Color = rate determines hue band
                pt.freqBand = std::clamp((std::abs(g.rate) - 0.25f) / 3.75f, 0.0f, 1.0f);
            } else {
                // Grain inactive — fade particle out
                pt.life -= 0.05f * freezeDamp;
                if (pt.life <= 0.0f) {
                    // Hidden off-screen until reactivated
                    pt.x = -100.0f;
                    pt.y = -100.0f;
                    pt.currentSize = 0.0f;
                }
            }
        }

        repaint();
    }

private:
    ZeroGrainProcessor& proc;

    CleanSliderLookAndFeel cleanLaf;

    juce::Slider densityKnob, pitchKnob, stretchKnob, reverseKnob, mixKnob;
    juce::Slider sizeKnob, scatterKnob, freezeKnob, focusKnob;
    juce::OwnedArray<juce::SliderParameterAttachment> attachments;

    int vizY = 0;
    int vizH = 200;

    struct Particle {
        float x, y, vx, vy, size, currentSize, life, phase, freqBand, homeY;
    };
    static constexpr int NUM_PARTICLES = GranularSynth::MAX_GRAINS;
    std::array<Particle, NUM_PARTICLES> particles{};

    void resetParticle(Particle& pt, bool randomPos) {
        float w = 536.0f;
        float h = 220.0f;
        if (randomPos) {
            pt.x = (float)(std::rand() % (int)w);
            pt.y = (float)(std::rand() % (int)h);
        }
        pt.vx = ((float)(std::rand() % 1000) / 1000.0f - 0.5f) * 2.0f;
        pt.vy = ((float)(std::rand() % 1000) / 1000.0f - 0.5f) * 2.0f;
        pt.size = 1.5f + (float)(std::rand() % 1000) / 1000.0f * 3.5f;
        pt.currentSize = pt.size;
        pt.life = 0.5f + (float)(std::rand() % 1000) / 1000.0f * 0.5f;
        pt.phase = (float)(std::rand() % 1000) / 1000.0f * 6.28f;
        pt.freqBand = (float)(std::rand() % 1000) / 1000.0f;
    }

    void drawParticles(juce::Graphics& g) {
        float freeze = *proc.apvts.getRawParameterValue("freeze");
        float freezeFactor = (freeze > 0.5f) ? (freeze - 0.5f) * 2.0f : 0.0f;

        for (auto& pt : particles) {
            float alpha = std::clamp(pt.life, 0.0f, 1.0f);
            if (alpha < 0.02f || pt.currentSize < 0.1f) continue;

            float px = pt.x;
            float py = pt.y;
            float sz = pt.currentSize;

            juce::Colour c;
            if (freeze > 0.5f) {
                float hue = pt.freqBand * 0.5f + 0.55f;
                juce::Colour warm = juce::Colour::fromHSV(hue, 0.5f, 0.7f, 1.0f);
                juce::Colour icy = juce::Colour::fromHSV(0.58f, 0.1f, 0.9f, 1.0f);
                c = warm.interpolatedWith(icy, freezeFactor);
            } else {
                float hue = pt.freqBand * 0.5f + 0.55f;
                c = juce::Colour::fromHSV(hue, 0.6f, 0.7f, 1.0f);
            }

            float glowSize = freeze > 0.5f ? sz * 4.0f : sz * 3.0f;
            if (sz > 1.5f && alpha > 0.06f) {
                g.setColour(c.withAlpha(alpha * 0.05f));
                g.fillEllipse(px - glowSize, py - glowSize, glowSize * 2.0f, glowSize * 2.0f);
            }

            g.setColour(c.withAlpha(alpha));
            g.fillEllipse(px - sz * 0.5f, py - sz * 0.5f, sz, sz);

            g.setColour(c.withAlpha(alpha * 0.2f));
            g.drawEllipse(px - sz * 0.5f, py - sz * 0.5f, sz, sz, 0.5f);
        }
    }

    void drawWaveform(juce::Graphics& g) {
        float w = (float)(getWidth() - 24);
        float h = (float)vizH;
        float x0 = 12.0f;
        float y0 = (float)vizY;

        int numPoints = (int)w;
        if (numPoints < 2) return;

        float level = proc.audioLevel.load(std::memory_order_relaxed);
        float audioEnergy = std::clamp(level * 8.0f, 0.0f, 1.0f);

        // Center line
        g.setColour(juce::Colour(0xffe0e0e0));
        g.drawLine(0.0f, y0 + h * 0.5f, x0 + w, y0 + h * 0.5f, 0.5f);

        int bufSize = proc.getSynth().getCircBufSize();
        if (bufSize < 512) return;

        // Shape from real input signal
        juce::Path wave;
        int readLen = std::min(numPoints, bufSize / 3);
        readLen = std::max(readLen, 128);

        for (int i = 0; i < numPoints; ++i) {
            float t = (float)i / (float)numPoints;
            int offset = (int)(t * (float)readLen);
            offset = std::clamp(offset, 0, bufSize - 2);
            float sample = proc.getSynth().getCircBufSample(offset);

            float yPos = y0 + h * 0.5f - sample * h * 0.45f;
            if (i == 0) wave.startNewSubPath(x0, yPos);
            else wave.lineTo(x0 + (float)i, yPos);
        }

        // Fill under waveform
        juce::Path fillPath(wave);
        fillPath.lineTo(x0 + w, y0 + h * 0.5f);
        fillPath.lineTo(x0, y0 + h * 0.5f);
        fillPath.closeSubPath();
        g.setColour(juce::Colour::fromRGBA(0, 153, 255, 25));
        g.fillPath(fillPath);

        // Light grey line
        g.setColour(juce::Colour(0xffccdde8));
        g.strokePath(wave, juce::PathStrokeType(1.2f));
    }

    juce::Colour getClusterColour(int index) {
        float hue = 0.6f - (index / 7.0f) * 0.5f;
        return juce::Colour::fromHSV(hue, 0.7f, 0.55f, 1.0f);
    }

    void drawClusterBars(juce::Graphics& g) {
        float barW = 16.0f;
        float barMaxH = 80.0f;
        float spacing = 48.0f;
        float startX = 36.0f;
        float y0 = (float)(getHeight() - 36);

        float level = proc.audioLevel.load(std::memory_order_relaxed);
        float audioEnergy = std::clamp(level * 8.0f, 0.0f, 1.0f);

        int maxCount = 1;
        for (int c = 0; c < 7; ++c) {
            int cnt = proc.getSynth().getClusterCount(c);
            if (cnt > maxCount) maxCount = cnt;
        }

        for (int c = 0; c < 7; ++c) {
            int cnt = proc.getSynth().getClusterCount(c);
            float normalized = (float)cnt / (float)maxCount;
            float filledH = barMaxH * std::clamp(normalized, 0.0f, 1.0f);

            float x = startX + c * spacing;

            juce::Colour col = getClusterColour(c);
            g.setColour(col.withAlpha(0.15f));
            g.fillRoundedRectangle(x, y0 - barMaxH, barW, barMaxH, 2.0f);
            g.setColour(col);
            g.fillRoundedRectangle(x, y0 - filledH, barW, filledH, 2.0f);
        }

        // Centroid marker — fades out when no sound
        float markerAlpha = std::clamp(audioEnergy * 3.0f, 0.0f, 1.0f);
        if (markerAlpha > 0.01f) {
            float centroid = proc.getSynth().getCurrentCentroid();
            float markerX = startX + centroid * 6.0f * spacing + barW * 0.5f - 1.0f;
            g.setColour(juce::Colour::fromRGBA(51, 51, 51, (juce::uint8)(markerAlpha * 200)));
            g.drawLine(markerX, y0 - barMaxH - 10, markerX, y0 + 2, 1.0f);
        }

        g.setColour(juce::Colour(0xff555555));
        g.setFont(juce::Font(9.0f, juce::Font::bold));
        const char* labels[] = {"SUB", "BASS", "LO-M", "MID", "HI-M", "HIGH", "AIR"};
        for (int c = 0; c < 7; ++c)
            g.drawText(labels[c], (int)(startX + c * spacing - 4), (int)(y0 + 4), (int)(barW + 8), 14, juce::Justification::centred);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ZeroGrainEditor)
};

