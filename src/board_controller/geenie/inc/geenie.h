#pragma once

#include <thread>

// #include "board.h"
// #include "board_controller.h"
// #include "serial.h"
// #include "geenie_gain_tracker.h"

#include <condition_variable>
#include <mutex>
#include <thread>

#include "bt_lib_board.h"
#include "geenie_gain_tracker.h"

class Geenie : public BTLibBoard
{

// protected:
//     GeenieGainTracker gain_tracker;
//     volatile bool keep_alive;
//     bool initialized;
//     bool is_streaming;
//     std::thread streaming_thread;
//     Serial *serial;

//     int open_port ();
//     int set_port_settings ();
//     void read_thread ();

// public:
//     Geenie (struct BrainFlowInputParams params);
//     ~Geenie ();

//     int prepare_session ();
//     int start_stream (int buffer_size, const char *streamer_params);
//     int stop_stream ();
//     int release_session ();
//     int config_board (std::string config, std::string &response);

//     static constexpr int start_byte = 0xA0;
//     static constexpr int end_byte = 0xC0;
//     static constexpr double ads_gain = 8.0;
//     static constexpr double ads_vref = 2.5;
// };

protected:
    GeenieGainTracker gain_tracker;
    volatile bool keep_alive;
    bool is_streaming;
    std::thread streaming_thread;
    std::mutex m;
    std::condition_variable cv;
    volatile int state;

    void read_thread ();
    std::string get_name_selector ();

public:
    Geenie (struct BrainFlowInputParams params);
    ~Geenie ();

    int prepare_session ();
    int start_stream (int buffer_size, const char *streamer_params);
    int stop_stream ();
    int release_session ();
    int config_board (std::string config, std::string &response);

    //     static constexpr int start_byte = 0xA0;
//     static constexpr int end_byte = 0xC0;
//     static constexpr double ads_gain = 8.0;
//     static constexpr double ads_vref = 2.5;
};
