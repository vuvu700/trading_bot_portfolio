import os
from holo import patternValidation


def rename(filename):
    # what should be modified
    match, datas = patternValidation(filename, "msg_prices_<platform>_<currency>_<timestamp>_<dateStart:s>_<houreStart:s>_<dateEnd:s>_<houreEnd:s>.json")
    # what is correct (and so shouldn't be modified)
    match2, _ = patternValidation(filename, "msg_prices_<platform>-<currency>_<timestamp>_<dateStart:s>_<houreStart:s>__<dateEnd:s>_<houreEnd:s>.json")
    # if (should be modified) and not (is correct)
    if match and not match2: # => rename it
        os.rename(filename, f"msg_prices_{datas['platform']}-{datas['currency']}_{datas['timestamp']}_{datas['dateStart']}_{datas['houreStart']}__{datas['dateEnd']}_{datas['houreEnd']}.json")
        return True
    return False

def match(filename):
    match, _ = patternValidation(filename, "msg_prices_<platform>-<currency>_<timestamp>_<dateStart:s>_<houreStart:s>_<dateEnd:s>_<houreEnd:s>.json")
    return match

#sum(rename(f) for f in os.listdir())
#sum(match(f) for f in os.listdir())