#include <iostream>
#include <csignal>
#include "LOG.h"
#include "Device.h"
#include "ProcessImage.h"

#include <getopt.h>

Device device;
ProcessImage processImage;

//static const char short_options[] = "c:d:f:F:hmo:rs:uz";
static const char short_options[] = "d:g:s:h";

static const struct option
        long_options[] = {
        {"device",  required_argument, NULL, 'd'},
        {"gain",    required_argument, NULL, 'g'},
        {"shutter", required_argument, NULL, 's'},
        {"help",    no_argument,       NULL, 'h'},
        {0, 0, 0,                            0}
};


std::string devName = "/dev/video0";
int gain = 50;
int shutter = 5000;


void usage(int argc, char **argv) {
    std::cout <<
              "Usage: " << argv[0] << "[options]\n\n"
                                      "Options:\n"
                                      "-d | --device name \tVideo device name (default: " << devName << ")\n"
                                                                                                        "-g | --gain value \tSet gain (default: "
              << gain << ", min: 1, max 480)\n"
                         "-s | --shutter value \tSet shutter (default: " << shutter << ", min: 29, max 15110711)\n"
                                                                                       "-h | --help \t\tPrint help\n";


}

void getopt(int argc, char **argv) {
    for (;;) {
        int index;
        int c;

        c = getopt_long(argc, argv,
                        short_options,
                        long_options,
                        &index);

        if (-1 == c)
            break;

        switch (c) {
            case 0: /* getopt_long() flag */
                break;

            case 'd':
                devName = optarg;
                break;

            case 'g':
                gain = atoi(optarg);
                break;

            case 's':
                shutter = atoi(optarg);
                break;

            case 'h':
                usage(argc, argv);
                exit(EXIT_SUCCESS);

            default:
                usage(argc, argv);
                exit(EXIT_FAILURE);
        }
    }
}


void SigintHandler(int dummy) {
    LOG_INFO("Stop device");
    device.Stop();
    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
    LOG_INFO("Start device");
    signal(SIGINT, SigintHandler);

    getopt(argc, argv);

    device.Init(devName);

    processImage.Init("Imx296");
    device.SetCallProcessImage(processImage.Processing);
    device.SetCallDrawImage(processImage.DrawImage);

    device.SetGain(gain);
    device.SetShutter(shutter);

    device.Start();
    device.Join();

    return 0;
}
