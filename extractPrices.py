if __name__ not in ("__main__", "__mp_main__"):
    raise ImportError(f"this script must be launched, not imported by {__name__!r}")
# => __name__ == __(mp_)main__

from pathlib import Path

from scraperLib.extract import (
    _PathLike, _FileFormat, 
    DEFAULT_FORMAT, DIRECTORY_MESSAGES,
    multipleExtractAndSaveParallel, )

from holo.__typing import get_args, assertIterableSubType, cast
from holo.prettyFormats import print_exception, prettyPrint, prettyTime

if __name__ == "__main__":
    try:
        from argparse import ArgumentParser
        
        parser = ArgumentParser(__file__)
        parser.add_argument("files", nargs='*', type=Path)
        parser.add_argument(
            "--format", action="store", default=DEFAULT_FORMAT, choices=get_args(_FileFormat), 
            help=f"the format of files to convert from msgs (default: {DEFAULT_FORMAT})")
        args = parser.parse_args()
        
        filesPaths:"list[_PathLike]|None" = assertIterableSubType(Path, args.files)
        if len(filesPaths) == 0: # => not given as argument
            filesPaths = None # => do for all files

        
        multipleExtractAndSaveParallel(
            filesPaths, DIRECTORY_MESSAGES, None, 
            saveFormat=cast(_FileFormat, args.format),
            overwrite=True, nbWorkers=12, batchSize=150, verbose=1)
        
    except Exception as err: print_exception(err)
    finally: input("press enter to exit ...")