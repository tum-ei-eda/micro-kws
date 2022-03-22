#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.gridspec as gridspec
import argparse
import serial
import time
import sys
import os
import threading
import warnings
import scipy.io.wavfile

# Ignore matplotlib warnings
warnings.filterwarnings('ignore')

UART_PACKET_FOOTER = b'\x00\x01\x02\x03\x04\x05\x06\x07'

class LoopRunner():

    def __init__(self, args: argparse.Namespace, ser: serial.Serial):
        self.args = args  # arguments from argparse
        self.ser = ser  # serial port object

        self.start_time = time.perf_counter()  # get start time of plotter
        self.animation = None  # animation object used by matplotlib
        self.run = False
        self.threads = []  # list of threads
        # The screen needs to be updated from the main Thread on MacOS
        # self.run_plotter_thread = threading.Thread(target=self.run_plotter)
        # self.threads.append(self.run_plotter_thread)
        self.run_loop_thread = threading.Thread(target=self.run_loop)
        self.threads.append(self.run_loop_thread)

        ##########################  Packet  ####################################
        #   Feature Data     Labels    Top Index     Footer                    #
        #    #########   +   #####   +   #####   +   ######                    #
        #   1960 Byte        4 Byte     1 Byte       8 Byte                    #
        ########################################################################

        self.feature_size = self.args.feature_height * self.args.feature_width
        self.feature_data = np.zeros(
            (self.args.feature_width, self.args.feature_height))
        self.feature_data.setflags(write=1)

        self.labels = self.args.standard_labels + self.args.category_labels

        self.top_category_index = 0

        self.packet_footer = UART_PACKET_FOOTER
        self.packet_size = self.feature_size + \
            len(self.labels) + 1 + len(self.packet_footer)

        self.category_array = np.zeros((2, len(self.labels)))
        self.time_array = [0]

    def start(self):
        self.run = True
        for x in self.threads:
            x.start()

    def stop(self):
        self.run = False
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()
        plt.close('all')
        for x in self.threads:
            x.join()

    def get_current_runtime(self):
        return round(time.perf_counter() - self.start_time, 4)

    def run_loop(self):

        while self.run:

            # TODO(fabianpedd): Do parsing differently and cleanup array indexing
            # Read data from serial port until footer and parse as numpy array
            raw_data = self.ser.read_until(self.packet_footer)
            data_as_int = np.frombuffer(raw_data, dtype=np.int8).copy()
            data_as_uint = np.frombuffer(raw_data, dtype=np.uint8).copy()

            # Check if data len machtes expected packet size
            if len(data_as_int) == self.packet_size:

                # Parse feature data
                self.feature_data = data_as_int[:-(self.packet_size - self.feature_size)
                                                ].reshape(self.args.feature_height, -1)
                self.feature_data = np.transpose(self.feature_data)

                # Parse category data and append to category array
                self.category_array = np.vstack(
                    (self.category_array, data_as_uint[self.feature_size:(self.feature_size + len(self.labels))]))

                self.top_category_index = data_as_uint[self.feature_size + len(
                    self.labels): self.feature_size + len(self.labels) + 1]
                # print(self.labels[int(self.top_category_index)])

                # Append time stamp to time array
                self.time_array.append(self.get_current_runtime())

            else:
                print('Wrong packet size:', data_as_int.size,
                      'but expected', self.packet_size)

    def run_plotter(self):

        # TODO(@PhilippvK): it would be cool if the colors in the debugger GPU
        # match the RGB values of the onboard LED
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2])
        fig.canvas.set_window_title('Micro-KWS Debugger')
        fig.set_facecolor('xkcd:grey')

        # Create feature graph
        ax1 = fig.add_subplot(gs[0, 0])
        feature_graph = ax1.imshow(
            self.feature_data, aspect='auto', cmap='gray', vmin=self.args.feature_min, vmax=self.args.feature_max)
        ax1.set_facecolor('xkcd:light grey')
        ax1.set_xlabel(r'Time bins')
        ax1.set_ylabel(r'Frequency / feature bins''\n''(<- higher | lower ->)')

        ax2 = fig.add_subplot(gs[0, 1])
        bar_plot = ax2.bar(range(len(
            self.labels)), self.category_array[-1], bottom=self.args.category_min, tick_label=self.labels, color=colors)
        ax2.set_xticklabels(self.labels, rotation=45, ha='right')
        ax2.set_ylim(self.args.category_min, self.args.category_max + 5)
        ax2.grid(axis='y', linewidth=0.5)

        # Create category history plot
        ax3 = fig.add_subplot(gs[1, :])
        category_graphs = []
        for i, l in enumerate(self.labels):
            category_graphs.append(
                ax3.plot([], [], color=colors[i], linewidth=1.5, label=l)[0])
        ax3.set_ylim(self.args.category_min - 5, self.args.category_max + 5)
        ax3.set_xlabel(r'Time [s]')
        ax3.set_ylabel(r'Confidence')
        labels = [l.get_label() for l in category_graphs]
        ax3.legend(category_graphs, labels, loc="upper left",
                   facecolor='xkcd:light grey')
        ax3.grid(linewidth=0.5)

        top_category_text = fig.text(
            0.1, 0.485, 'Detected:' + self.labels[int(self.top_category_index)], weight='semibold', size='large')

        plt.tight_layout()

        self.animation = anim.FuncAnimation(fig, self.plotter, fargs=(
            [feature_graph, bar_plot, category_graphs, top_category_text, ax1, ax2, ax3]), interval=100)

        # Start animation until window is closed
        plt.show()

        # When we return from show, end the simulation
        self.run = False

    def plotter(self, frame, feature_graph, bar_plot, category_graphs, top_category_text, ax1, ax2, ax3):

        # Set feature data
        feature_graph.set_data(self.feature_data)

        # Set bar plot category data
        for bar, d in zip(bar_plot, self.category_array[-1]):
            bar.set_height(d - self.args.category_min)

        # Determine how many past category values we can actually print
        plot_count = min(
            self.args.category_history, min(len(self.category_array), len(self.time_array)))
        # and set limits accordingly
        ax3.set_xlim(min(self.time_array[-plot_count:], default=0),
                     max(self.time_array[-plot_count:], default=0))

        # Set category data
        for i, g in enumerate(category_graphs):
            g.set_data(self.time_array[-plot_count:],
                       self.category_array[-plot_count:, i])

        top_category_text.set_text('Detected: %s' %
                                   self.labels[int(self.top_category_index)])


def main():
    parser = argparse.ArgumentParser(description='UART utility.')

    # UART parameters
    parser.add_argument('-p', '--port', type=str,
                        default='/dev/ttyUSB0', help='The path to the UART device.')
    parser.add_argument('-b', '--baudrate', type=int,
                        default=200000, help='The baudrate of the UART device.')
    parser.add_argument('-pf', '--packet_footer', type=str,
                        default='\x00\x01\x02\x03\x04\x05\x06\x07', help='The footer of each UART packet.')

    # Feature data parameters
    parser.add_argument('-fh', '--feature-height', type=float, default=49,
                        help='Height of feature image (aka. feature_slice_size).')
    parser.add_argument('-fw', '--feature-width', type=float, default=40,
                        help='Width of feature image (aka. feature_slize_count).')
    parser.add_argument('-fmin', '--feature-min', type=float,
                        default=-128, help='Minimum value a feature can have.')
    parser.add_argument('-fmax', '--feature-max', type=float,
                        default=127, help='Maximum value a feature can have.')

    # Category data parameters
    parser.add_argument('-sl', '--standard-labels', nargs='+', default=[
                        'silence', 'unknown'], help='The two standard labels of (almost) all KWS models.')
    parser.add_argument('-cl', '--category-labels', nargs='+', default=[
                        'yes', 'no'], help='The actual category labels of the model, excluding silence and unknown.')
    parser.add_argument('-cmin', '--category-min', type=float,
                        default=0, help='Minimum value a category can have.')
    parser.add_argument('-cmax', '--category-max', type=float,
                        default=255, help='Maximum value a category can have.')

    # Plotting parameters
    parser.add_argument('-ch', '--category-history', type=float,
                        default=20, help='How many past category values to show.')

    # Audio recording parameters
    parser.add_argument('-a', '--audio', default=False,
                        action='store_true', help='Activate audio mode.')
    parser.add_argument('-atl', '--audio-total-length', type=float,
                        default=2.0, help='How long the total audio recording will be [in seconds].')
    parser.add_argument('-apl', '--audio-packet-length', type=float,
                        default=0.1, help='Length of each audio packet transmitted via the serial connection [in seconds].')
    parser.add_argument('-afn', '--audio-file-name', type=str,
                        default='audio_test.wav', help='Name of the audio file to create (needs to end in .wav).')

    # Parse arguments
    args = parser.parse_args()

    # Initalize UART connection
    print('Starting UART connection on %s with %d baud.' %
          (args.port, args.baudrate))
    ser = serial.Serial(args.port, args.baudrate, timeout=0.75)
    ser.reset_input_buffer()

    if args.audio:
        audio_buffer = np.array([], dtype=np.int16)
        audio_received_bytes = 0
        audio_total_bytes = int(2 * 16000 * args.audio_total_length)
        audio_payload_bytes = int(2 * 16000 * args.audio_packet_length)
        audio_packet_bytes = audio_payload_bytes + len(UART_PACKET_FOOTER)
        last_packet_time = 0
        print('Waiting for audio data...')
        while audio_received_bytes < audio_total_bytes and \
                (last_packet_time == 0 or
                 time.perf_counter() - last_packet_time < 2):
            raw_data = ser.read_until(UART_PACKET_FOOTER)
            if len(raw_data) != audio_packet_bytes:
                print("Received", len(raw_data),
                      'bytes. Waiting for audio data...')
                continue
            last_packet_time = time.perf_counter()
            audio_data = np.frombuffer(
                raw_data[:audio_payload_bytes], dtype=np.int16).copy()
            audio_buffer = np.append(audio_buffer, audio_data)
            # TODO(fabianpedd): Use getsizeof instead
            audio_received_bytes += len(audio_data) * 2
            print('Received', len(audio_data) * 2, 'bytes,',
                  audio_total_bytes - audio_received_bytes, 'bytes remaining...')

        if audio_received_bytes < audio_total_bytes:
            print('Received only', audio_received_bytes, 'of total',
                  audio_total_bytes, 'bytes. But quit receiving prematurely due to time reasons.')
        else:
            print('Received full', audio_received_bytes, 'bytes.')

        scipy.io.wavfile.write(args.audio_file_name, 16000, audio_buffer)
        print('Wrote', audio_received_bytes, 'bytes to', args.audio_file_name)

        sys.exit(0)

    # Create loop runner object
    lr = LoopRunner(args, ser)

    # Try / catch in order to be able to kill all running threads on CTRL+C
    try:
        lr.start()

        while lr.run:
            lr.run_plotter()

        raise KeyboardInterrupt

    except KeyboardInterrupt as e:
        lr.stop()
        ser.close()
        print('Exiting program ...')
        os._exit(0) # hard exit, sys.exit(0) gets hung up


if __name__ == '__main__':
    main()
