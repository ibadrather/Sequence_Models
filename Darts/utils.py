# fix pythonpath when working locally
import sys
import os
from os import getcwd
from os.path import basename, dirname


def fix_pythonpath_if_working_locally():
    """Add the parent path to pythonpath if current working dir is darts/examples"""
    cwd = getcwd()
    if basename(cwd) == "examples":
        sys.path.insert(0, dirname(cwd))

    
def clear_terminal():
    # CLear terminal
    try:
        os.system("clear")
    except:
        pass