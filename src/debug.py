import logging as lg
import sys

def init_log(stream, mode):
    log_level = {
        'DEBUG': lg.DEBUG,
        'INFO': lg.INFO,
        'WARNING': lg.WARNING,
        'ERROR': lg.ERROR,
        'CRITICAL': lg.CRITICAL
    }.get(mode, lg.WARNING)

    lg.basicConfig(stream=sys.stderr, level=log_level)

def log(string):
    lg.debug(string)

#lg.basicConfig(stream=sys.stderr, level=lg.DEBUG)
