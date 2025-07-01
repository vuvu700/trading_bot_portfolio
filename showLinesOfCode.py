from pathlib import Path
import re

from holo.files import (
    get_subdirectorys, get_subfiles, 
    countLines_singleFile, PATTERN_PY_FILES, )
from holo.prettyFormats import _ObjectRepr, prettyPrint, PrettyPrint_CompactArgs

###########################################################################
## just change "countLines_singleFile" in order to mesure something else ## 
########################################################################### 

class Tree():
    BASE_INDENT = ("|" + " " * 3)
    
    def __init__(self, path:Path, filesPattern:"re.Pattern") -> None:    
        self.path: Path = path; del path
        self.files: "list[tuple[Path, int]]" = []
        self.subTrees: "list[Tree]" = []
        self.__cachedSize: "int|None" = None
        # create reccursively
        for filename in get_subfiles(self.path):
            if filesPattern.fullmatch(filename) is None:
                continue # => didn't matched the pattern, not an interesting file
            filepath = self.path.joinpath(filename)
            self.files.append((filepath, countLines_singleFile(filepath)))
        for dirname in get_subdirectorys(self.path):
            self.subTrees.append(Tree(self.path.joinpath(dirname), filesPattern))
    
    def size(self)->int:
        if self.__cachedSize is None:
            self.__cachedSize = sum(size for _, size in self.files)
            self.__cachedSize += sum(map(Tree.size, self.subTrees))
        return self.__cachedSize
    
    def show(self, indentLevel:int=0)->None:
        if self.size() == 0: return
        indent = self.BASE_INDENT * indentLevel
        print(f"{indent}> {self.path.name} [total: {self.size():_d} lines]")
        for file, size in sorted(self.files, key=lambda t:t[1], reverse=True):
            print(f"{indent}{self.BASE_INDENT}- [{size:_d} lines] {file.name}")
        for tree in sorted(self.subTrees, key=lambda t:t.size(), reverse=True):
            tree.show(indentLevel+1)
    
    def __pretty__(self, *_, **__) -> _ObjectRepr:
        kwargs = {
            "path": self.path, "size": f"{self.size():_d} LOC", 
            "files": sorted(((f"{size:_d} LOC", path.as_posix()) for path, size in self.files), 
                            reverse=True, key=lambda s:int(s[0].split(" ")[0]))}
        filteredDirs = [tree for tree in self.subTrees if tree.size() != 0]
        if len(filteredDirs) != 0:
            kwargs["dirs"] = sorted(filteredDirs, key=lambda t:t.size(), reverse=True)
        return _ObjectRepr(self.__class__.__name__, (), kwargs)
    

if __name__ == "__main__":
    tree = Tree(Path("."), re.compile(PATTERN_PY_FILES))
    #prettyPrint(tree, specificCompact={tuple}, 
    #    compact=PrettyPrint_CompactArgs(compactSmaller=False, compactLarger=False))
    tree.show()
    