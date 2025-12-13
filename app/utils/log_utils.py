import sys
import logging

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
    

def print_log(print_string, log):
    """Prints the message to the console and writes it to the log file."""
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()