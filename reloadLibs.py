import sys, pathlib, paths_cfg, _frozen_importlib_external

def clear_imports_in_dir(target_dir: str, keeps:"list[str]|None"=None, verbose:bool=False):
    if keeps is None: keeps = []
    toDel: list[str] = []
    pTarget = paths_cfg.CURRENT_DIRECTORY.joinpath(target_dir).absolute().as_posix()
    if verbose: print(pTarget)
    
    for name, module in list(sys.modules.items()):
        module_file: "str|None" = getattr(module, "__file__", None)
        if module_file is None:
            if hasattr(module, "__spec__") and (getattr(module.__spec__, "origin", None) == 'built-in'):
                continue
            if (getattr(module, "__package__", None) is None) or (hasattr(module, "__loader__") is False):
                continue
            if type(module.__loader__) is type:
                continue
            if isinstance(module.__loader__, _frozen_importlib_external._NamespaceLoader):
                paths: "list[str]" = module.__loader__._path._path # type: ignore
                #print(f"--- {name} -> {paths}")
                for path in paths:
                    path = pathlib.Path(path).as_posix()
                    if path.startswith(pTarget):
                        toDel.append(name)
                        if verbose: print(f"to unload (weird): {name}")
                        break
                else: # no break
                    continue
        
        if module_file is not None:
            pModule = pathlib.Path(module_file).absolute().as_posix()
            if pModule.startswith(pTarget):
                toDel.append(name)
    
    for name in toDel:
        if any(name.startswith(k) for k in keeps):
            if verbose: print(f"kept: {name}")
            continue
        del sys.modules[name]
        if verbose: print(f"Unloaded {name}")

def exemple():
    print(f"nb libs before: {len(sys.modules)}")
    clear_imports_in_dir(".", keeps=["calculationLib"], verbose=True)
    print(f"nb libs after: {len(sys.modules)}")