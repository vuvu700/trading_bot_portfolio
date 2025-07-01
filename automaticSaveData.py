if __name__ not in ("__main__", "__mp_main__"):
    raise ImportError(f"this script must be launched, not imported by {__name__!r}")
# => __name__ == __(mp_)main__

if __name__ == "__main__":
    print("script is starting up, please wait ...")

if __name__ == "__main__":
    try:
        from holo.prettyFormats import prettyTime, print_exception
        import termcolor, colorama
        colorama.init()
    except Exception as err: # the holo lib might have an error
        print(f"{type(err).__name__}: {err}")
        input("press enter to exit ...")
        exit(-1)

    try:
        from datetime import datetime
        from argparse import ArgumentParser

        from scraperLib import downloadDatas 
        from scraperLib.dataScrapingConfig import (
            CONFIG as DEFAULT_SYMBOLS_CONFIG, SymbolConfig,
        )
        from data.get_data import DEFAULT_FORMAT
        
        import mergeDatas
        import check_for_datasets_splits
        
        from holo.__typing import NamedTuple



        class Args(NamedTuple):
            symbolsToDo:"list[SymbolConfig]"
            noMinPeriodes: bool
            downloadMaxDatas: bool
            dontMerge: bool
            dontCheckSplits: bool
            exit: bool

        argPraser = ArgumentParser(__file__)
        argPraser.add_argument("symbolsToDo", nargs='*', type=SymbolConfig.createFromText,
                            help=("if given gather specificly the symbols as '<platformeName>_<symbol>_<timeFrame>', "
                                    + "if not given: use the default config of symbols"))
        argPraser.add_argument("-f", action="store_true", dest="noMinPeriodes", default=False, required=False,
                            help="to force the gathering some datas for all targets")
        argPraser.add_argument("--downloadMaxDatas", action="store_true", default=False, required=False,
                            help="download the maximum amount of data")
        argPraser.add_argument("--dontMerge", action="store_true", default=False, required=False,
                            help="will not merge the new datas to the old prices datas")
        argPraser.add_argument("--dontCheckSplits", action="store_true", default=False, required=False,
                            help="will not check the number of splits")
        argPraser.add_argument("--exit", action="store_true", default=False, required=False,
                            help="will not ask to press enter when finished (great for scripts)")
        # TODO: add arg to don't merge the datas and one to not do the compleat checking of splits
        args = Args(**argPraser.parse_args().__dict__)

        # determine what symbols to do
        symbolsToDo: "list[SymbolConfig]" = args.symbolsToDo
        if len(args.symbolsToDo) == 0:
            symbolsToDo = DEFAULT_SYMBOLS_CONFIG

        # start the hard work
        tStart = datetime.now()
        manager = downloadDatas.Downloader(
            symbolsToDo, noMinPeriodes=args.noMinPeriodes,
            downloadMaxDatas=args.downloadMaxDatas)
        manager.start()
        manager.join()
        # => finished all the symbols

        print(f"total time taken to download and transforme the new datas: {prettyTime(datetime.now() - tStart)}\n")

        # tell the user the status of what havent been done
        errorHappened: bool = (len(manager.failedkeys) != 0)
        if errorHappened is True:
            termcolor.cprint(
                text=f"the following keys have failed to download: {manager.failedkeys}\n",
                color="red")

        if len(manager.notDownloadedKeys) != 0:
            termcolor.cprint(
                text=("the following keys have not been processed"
                    + f" (not enought to download): {manager.notDownloadedKeys}\n"),
                color="light_yellow")
        
        hasDownloaded: bool = (len(manager.downloadedKeys) >= 1)
        if (hasDownloaded is True):
            termcolor.cprint(
                text=f"the following keys have been downloaded: {manager.downloadedKeys}\n",
                color="light_blue")

        # merge the datas
        if (args.dontMerge is False) and (hasDownloaded is True):
            # => asked to merge
            mergeDatas.main(
                fileFormat=DEFAULT_FORMAT, removeOldFiles=True, parallel=True,
                keysToMerge=set(manager.downloadedKeys), verbose=1)
            print()
        
        # check the splits
        if (args.dontCheckSplits is False) and (hasDownloaded is True):
            # => asked to check for splits
            check_for_datasets_splits.main(
                showAll=False, simpleDateFromat=False, 
                showTotalSize=False, fileFormat=DEFAULT_FORMAT)
            print()
            
        totalTime = (datetime.now() - tStart)
        if errorHappened is False:
            termcolor.cprint(
                text=f"sucessfully finished everything in {prettyTime(totalTime)}",
                color="green")
        else: termcolor.cprint(
                text=f"something has failed, everything done in {prettyTime(totalTime)}",
                color="red")
        print()
        
    except Exception as err:
        print(colorama.Fore.RED)
        print_exception(err)
        print(colorama.Fore.RESET)

    finally: # => the program will never close by itself
        askExit: bool = True
        try: askExit = (not args.exit) # type: ignore => args migth not exist but is okay
        except: pass
        if askExit:
            input("press enter to exit ...")
