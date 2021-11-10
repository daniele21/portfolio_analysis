import os
from typing import Text
import logging

logger = logging.getLogger('OS manager')


def ensure_folder(folder: Text):
    if not os.path.exists(folder):
        os.mkdir(folder)
        logger.info(f' > Directory created at {folder}')
        return True
    else:
        logger.info(f' > Directory found at {folder}')
        return False
