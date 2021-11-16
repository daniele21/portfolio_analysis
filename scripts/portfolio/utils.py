import logging
import re
from typing import Text

logger = logging.getLogger('Check')


def check_date(date: Text):
    date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})', date)
    if not date_match:
        error = f' > No valid date format. Give YYYY-MM-DD'
        logger.error(error)
        return False
    else:
        return True
