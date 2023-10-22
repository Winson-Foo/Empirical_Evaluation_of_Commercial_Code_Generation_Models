from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import collections
import signal
import logging

import numpy as np

class ProgressBar(object):
    """Displays a progress bar.
    
    Parameters
    ----------
    target : int, optional
        Total number of steps expected, None if unknown.
    width : int, optional
        Progress bar width on screen.
    verbose : int, optional (default=1)
        Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    stateful_metrics : iterable of str, optional
        Iterable of string names of metrics that should *not* be averaged over time. 
        Metrics in this list will be displayed as-is. All others will be averaged 
        by the progress bar before display.
    interval : float, optional (default=0.05)
        Minimum visual progress update interval (in seconds).
    """

    ETA_THRESHOLD = 3600
    MS_PER_STEP = 1e-3
    US_PER_STEP = 1e-6

    def __init__(self, target=None, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.stateful_metrics = set(stateful_metrics or [])
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        
        Parameters
        ----------
        current : int
            Index of current step.
        values : list of tuple
            List of tuples `(name, value_for_last_step)`. If `name` is in `stateful_metrics`,
            `value_for_last_step` will be displayed as-is. Else, an average of the metric over
            time will be displayed.
        """
        values = values or []
        self._update_values(current, values)
        now = time.time()

        if self.verbose == 1:
            if self._should_update(now, current):
                self._print_bar(current)
                self._print_info(now)
                self._last_update = now

        elif self.verbose == 2:
            if self._should_update(now, current) or current >= self.target:
                self._print_info(now)

        sys.stdout.flush()

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def _update_values(self, current, values):
        for name, value in values:
            if name not in self.stateful_metrics:
                if name not in self._values:
                    self._values[name] = [value * (current - self._seen_so_far),
                                          current - self._seen_so_far]
                else:
                    self._values[name][0] += value * (current - self._seen_so_far)
                    self._values[name][1] += current - self._seen_so_far
            else:
                self._values[name] = [value, 1]
        self._seen_so_far = current

    def _should_update(self, now, current):
        return now - self._last_update >= self.interval and \
               self.target is not None and \
               current < self.target

    def _print_bar(self, current):
        prev_total_width = self._total_width
        if self._dynamic_display:
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')

        barstr = self._get_bar_string(current)
        sys.stdout.write(barstr)

        self._total_width = len(barstr)

        if prev_total_width > self._total_width:
            sys.stdout.write(' ' * (prev_total_width - self._total_width))
        elif prev_total_width < self._total_width:
            sys.stdout.write('\b' * (self._total_width - prev_total_width))

    def _get_bar_string(self, current):
        if self.target is not None:
            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%d [' % (numdigits, self.target)
            bar = barstr % current
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
        else:
            bar = '%7d/Unknown' % current
        
        return bar

    def _print_info(self, now):
        info = ' - %.0fs' % (now - self._start)

        if self.target is not None and self._seen_so_far < self.target:
            info += self._get_eta_string(now)

        for name, value in self._values.items():
            info += ' - %s:' % name
            if isinstance(value, list):
                info += ' %.4f' % (np.mean(value[0] / max(1, value[1])))
            else:
                info += ' %s' % value

        self._total_width += len(info)

        if self.verbose == 1 and self.target and self._seen_so_far >= self.target:
            info += '\n'

        sys.stdout.write(info)

    def _get_eta_string(self, now):
        time_per_step = (now - self._start) / self._seen_so_far
        if time_per_step >= 1:
            time_fmt = '%ds/step'
            time_per_step = int(round(time_per_step))
        elif time_per_step >= ProgressBar.MS_PER_STEP:
            time_fmt = '%.0fms/step'
            time_per_step *= ProgressBar.MS_PER_STEP
        else:
            time_fmt = '%.0fus/step'
            time_per_step *= ProgressBar.US_PER_STEP

        if time_per_step >= 1:
            eta = time_per_step * (self.target - self._seen_so_far)
            if eta >= ProgressBar.ETA_THRESHOLD:
                eta_fmt = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta >= 60:
                eta_fmt = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_fmt = '%ds' % eta

            eta_str = ' - ETA: %s' % eta_fmt
        else:
            eta_str = ''

        return time_fmt % time_per_step + eta_str


class DelayedInterrupt(object):
    '''
    class based on: http://stackoverflow.com/a/21919644/487556
    
    It delays interrupts until the code exits the entered block
    '''
    def __init__(self, signals):
        if not isinstance(signals, list) and not isinstance(signals, tuple):
            signals = [signals]
        self.sigs = signals        

    def __enter__(self):
        self.signal_received = {}
        self.old_handlers = {}
        for sig in self.sigs:
            self.signal_received[sig] = False
            self.old_handlers[sig] = signal.getsignal(sig)
            def handler(s, frame):
                self.signal_received[sig] = (s, frame)
                # Note: in Python 3.5, you can use signal.Signals(sig).name
            self.old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

    def __exit__(self, type, value, traceback):
        for sig in self.sigs:
            signal.signal(sig, self.old_handlers[sig])
            if self.signal_received[sig] and self.old_handlers[sig]:
                self.old_handlers[sig](*self.signal_received[sig])

class DelayedKeyboardInterrupt(DelayedInterrupt):
    def __init__(self):
        super(DelayedKeyboardInterrupt, self).__init__([signal.SIGINT])