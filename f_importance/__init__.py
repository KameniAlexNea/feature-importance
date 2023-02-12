__version__ = "1.1.1"

import logging

logging.basicConfig(
    filemode="test.log",
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%H:%M:%S",
)
