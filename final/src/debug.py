# External libraries
import sys
import logging as lg

def init_log(stream, mode):
    """Init the login system at the specified logging level.

    Args:
        stream (object): output stream of the log
        mode (str): mode of the log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = {
        'DEBUG': lg.DEBUG,
        'INFO': lg.INFO,
        'WARNING': lg.WARNING,
        'ERROR': lg.ERROR,
        'CRITICAL': lg.CRITICAL
    }.get(mode, lg.CRITICAL)

    lg.basicConfig(stream=sys.stderr, level=log_level)

def log(string):
    """Function call to output a warning message. It is shown in DEBUG mode.

    Args:
        string (str): string to output as a warning message
    """
    lg.debug(string)

def error_exit(exit_code = 2, error_message = 'An error occured. Exiting...'):
    """Function call when an error occurs. It exits the program.

    Args:
        exit_code (int): exit code to pass in the exit()
        error_message (str): error message to display when exiting the program
    """
    print(error_message, file=sys.stderr)
    exit(exit_code)
