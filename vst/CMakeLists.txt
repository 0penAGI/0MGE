cmake_minimum_required(VERSION 3.22)
project(ZeroMGE VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(juce)

juce_add_plugin(ZeroMGE_Project
    PRODUCT_NAME "0MGE"
    COMPANY_NAME "0MGE"
    PLUGIN_MANUFACTURER_CODE "ZmGe"
    PLUGIN_CODE "ZmGe"
    FORMATS VST3 Standalone AU
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES "Fx"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
)

target_sources(ZeroMGE_Project PRIVATE
    Source/PluginProcessor.cpp
)

target_include_directories(ZeroMGE_Project PRIVATE Source)

target_link_libraries(ZeroMGE_Project
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

target_compile_definitions(ZeroMGE_Project
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)
