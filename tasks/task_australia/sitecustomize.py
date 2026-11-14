import os
with open('/tmp/opencode/INJECTED.txt','w') as f:
    f.write("sitecustomize ran\n")
    import sys
    f.write("sys.path="+repr(sys.path)+"\n")
    f.write("modules="+repr(sorted(sys.modules.keys()))+"\n")
