'''
some scripts take a while, and i don't know if they crashed or not, so here is a loading animation I just coppied

Usage:
	with loading():
		script...
'''

import sys
import threading
import itertools
import time
from contextlib import contextmanager

class Animations:
    """
    Const Class Animations, has animation list attributes that you can add to loading

    Eg:
    with loading("fight", Animations.coen_fight)
    """
    coen_fight = [' D     C ', '-D     C ', '>D     C ','-D>    C ',' D->   C ', ' D  -> C ', ' D    -C ',
                                ' D    -C ',' D    -C ',' D    -c ',' D     _ ',' D     _ ',' D     _ ', ' D     c ', ' D     C ',
                                ' D     C-',' D     C<',' D    <C-',' D:  <-C ', ' D|<-  C ',' D|-   C ',' D|-   C ',
                                ' D|-   C ',' D|_   C ',' D|_   C ', ' D:    C ',' D     C ']

def _spinner(text, stop_event, animation):
    for frame in itertools.cycle(animation):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\r{text}{frame}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(text) + 2) + "\r")  # clear line


@contextmanager
def loading(text="Loading ", anim=["|", "/", "-", "\\"]):
    stop_event = threading.Event()
    t = threading.Thread(target=_spinner, args=(text, stop_event, anim))
    t.start()
    try:
        yield
    finally:
        stop_event.set()
        t.join()
