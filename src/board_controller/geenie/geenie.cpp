// #include <math.h>
// #include <string.h>
// #include <vector>

// #include "custom_cast.h"
// #include "geenie.h"
// #include "serial.h"
// #include "timestamp.h"

#include <string.h>

#include "geenie.h"

#include "custom_cast.h"
#include "get_dll_dir.h"
#include "timestamp.h"


// constexpr int Geenie::start_byte;
// constexpr int Geenie::end_byte;
// constexpr double Geenie::ads_gain;
// constexpr double Geenie::ads_vref;

#define START_BYTE 0x41
#define END_BYTE_STANDARD 0xC0
#define END_BYTE_MAX 0xC6


Geenie::Geenie (struct BrainFlowInputParams params)
    : BTLibBoard ((int)BoardIds::GEENIE_BOARD, params)
{
    keep_alive = false;
    state = (int)BrainFlowExitCodes::SYNC_TIMEOUT_ERROR;
}

Geenie::~Geenie ()
{
    skip_logs = true;
    release_session ();
}

int Geenie::prepare_session ()
{
    if (params.ip_port <= 0)
    {
        params.ip_port = 1; // default for enophone
    }
    return BTLibBoard::prepare_session ();
}

int Geenie::start_stream (int buffer_size, const char *streamer_params)
{
    if (!initialized)
    {
        safe_logger (spdlog::level::err, "You need to call prepare_session before start_stream");
        return (int)BrainFlowExitCodes::BOARD_NOT_CREATED_ERROR;
    }
    if (keep_alive)
    {
        safe_logger (spdlog::level::err, "Streaming thread already running");
        return (int)BrainFlowExitCodes::STREAM_ALREADY_RUN_ERROR;
    }

    int res = prepare_for_acquisition (buffer_size, streamer_params);
    if (res != (int)BrainFlowExitCodes::STATUS_OK)
    {
        return res;
    }

    res = bluetooth_open_device ();
    if (res != (int)BrainFlowExitCodes::STATUS_OK)
    {
        return res;
    }

    keep_alive = true;
    streaming_thread = std::thread ([this] { this->read_thread (); });
    // wait for the 1st package received
    std::unique_lock<std::mutex> lk (this->m);
    auto sec = std::chrono::seconds (1);
    int num_secs = 5;
    if (cv.wait_for (lk, num_secs * sec,
            [this] { return this->state != (int)BrainFlowExitCodes::SYNC_TIMEOUT_ERROR; }))
    {
        return state;
    }
    else
    {
        safe_logger (spdlog::level::err, "no data received in {} sec, stopping thread", num_secs);
        stop_stream ();
        return (int)BrainFlowExitCodes::BOARD_NOT_READY_ERROR;
    }
}

int Geenie::stop_stream ()
{
    if (keep_alive)
    {
        keep_alive = false;
        streaming_thread.join ();
        state = (int)BrainFlowExitCodes::SYNC_TIMEOUT_ERROR;
        return bluetooth_close_device ();
    }
    else
    {
        return (int)BrainFlowExitCodes::STREAM_THREAD_IS_NOT_RUNNING;
    }
}

int Geenie::release_session ()
{
    if (initialized)
    {
        stop_stream ();
        free_packages ();
    }
    return BTLibBoard::release_session ();
}

void Geenie::read_thread ()
{
    /*
        Byte 1: 0xA0
        Byte 2: Sample Number
        Bytes 3-5: Data value for EEG channel 1
        Bytes 6-8: Data value for EEG channel 2
        Bytes 9-11: Data value for EEG channel 3
        Bytes 12-14: Data value for EEG channel 4
        Bytes 15-17: Data value for EEG channel 5
        Bytes 18-20: Data value for EEG channel 6
        Bytes 21-23: Data value for EEG channel 6
        Bytes 24-26: Data value for EEG channel 8
        Aux Data Bytes 27-32: 6 bytes of data
        Byte 33: 0xCX where X is 0-F in hex
    */
    int res;
    bool is_ready = false;
    // unsigned char b[32];
    unsigned char b[30];
    // double accel[3] = {0.};
    int num_rows = board_descr["default"]["num_rows"];
    double *package = new double[num_rows];
    for (int i = 0; i < num_rows; i++)
    {
        package[i] = 0.0;
    }
    std::vector<int> eeg_channels = board_descr["default"]["eeg_channels"];
    // double accel_scale = (double)(0.002 / (pow (2, 4)));

    while (keep_alive)
    {
        // check start byte
        // TODO Does the converion to char * work???
        int res = bluetooth_get_data ((char *)b, 1);
        if ((res != 1) || (b[0] != START_BYTE))
        {
            continue;
        }
        is_ready = true;
        double timestamp = get_timestamp ();
        if (state != (int)BrainFlowExitCodes::STATUS_OK)
        {
            {
                std::lock_guard<std::mutex> lk (m);
                state = (int)BrainFlowExitCodes::STATUS_OK;
            }
            cv.notify_one ();
            safe_logger (spdlog::level::debug, "start streaming");
        }
        // res = serial->read_from_serial_port (b, 1);

        int remaining_bytes = 30;
        int pos = 0;

        if (is_ready)
        {
            while ((remaining_bytes > 0) && (keep_alive))
            {
                // TODO Does the converion to char * work???
                res = bluetooth_get_data ((char *)b + pos, remaining_bytes);
                remaining_bytes -= res;
                pos += res;
            }
            if (!keep_alive)
            {
                break;
            }

            if ((b[29] < END_BYTE_STANDARD) || (b[29] > END_BYTE_MAX))
            {
                safe_logger (spdlog::level::warn, "Wrong end byte {}", b[29]);
                continue;
            }

            // package num
            package[board_descr["default"]["package_num_channel"].get<int> ()] = (double)b[0];
            // eeg
            for (unsigned int i = 0; i < eeg_channels.size (); i++)
            {
                double eeg_scale = (double)(4.5 / float ((pow (2, 23) - 1)) /
                    gain_tracker.get_gain_for_channel (i) * 1000000.);
                package[eeg_channels[i]] = eeg_scale * cast_24bit_to_int32 (b + 1 + 3 * i);
            }

            package[board_descr["default"]["timestamp_channel"].get<int> ()] = get_timestamp ();

            push_package (package);
        }
    }
    delete[] package;
}


int Geenie::config_board (std::string config, std::string &response)
{
    safe_logger (spdlog::level::debug, "config_board is not supported for Enophone");
    return (int)BrainFlowExitCodes::UNSUPPORTED_BOARD_ERROR;
}

std::string Geenie::get_name_selector ()
{
    return "Geenie";
}