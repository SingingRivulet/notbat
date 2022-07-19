project("midirender")

#set(CMAKE_CXX_FLAGS "-std=c++20 -ggdb")
set(CMAKE_CXX_FLAGS "-std=c++2a -O3")

set(CMAKE_EXE_LINKER_FLAGS "-static")

include_directories(./)

add_executable(midirender 
    Binasc.cpp
    kalman.cpp
    MidiFile.cpp
    MidiMessage.cpp  
    Options.cpp
    synth.cpp
    editTable.cpp
    MidiEvent.cpp
    midiLoader.cpp
    midisplit.cpp
    player.cpp
    synthOutput.cpp
    hbb.cpp
    MidiEventList.cpp
    midiMap.cpp
    offline.cpp
    stringPool.cpp
    main.cpp)