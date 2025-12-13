import sys
import time
import logging


# ------------
# Logger Class
# ------------
class LoggerConfig:
    def __init__(self):
        self.log_level = "INFO"

    def get_logger(self, name: str) -> logging.Logger:
        level_int = getattr(logging, self.log_level.upper(), logging.INFO)
        
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.setLevel(level_int)
        logger.propagate = False
        
        return logger
    

# ------------------------
# Metrics Monitoring Class
# ------------------------
class AverageMeter(object):
    """Computes and stores the average and current value of metrics during training."""
    def __init__(self):
        self.reset()

    # Reset all statistics.
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # Update the statistics for the meter
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -------------------------
# Time Formatting Functions
# -------------------------
def time_string():
    """Generate a string that present current time"""
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string

def convert_secs2time(epoch_time):
    """Convert a time in seconds to a more readable format (hours, minutes, seconds)."""
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

def print_log(print_string, log):
    """Prints the message to the console and writes it to the log file."""
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()