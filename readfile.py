# -*- coding:UTF-8 -*-

def read(fp):
    l = []
    with open(fp, 'r') as f:
        for line in f:
            if not line.isspace():
                l.append(line)
    return l
