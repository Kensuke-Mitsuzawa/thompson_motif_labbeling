#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import collections

for line in sys.stdin:
    if line == 'SV\n':
        break

    weight = collections.defaultdict(lambda: 0)

    for line in sys.stdin:
        split = line.split()
        coef = float(split[0])
        for feature in split[1:]:
            number, count = map(int, feature.split(':'))
            weight[number] += coef * count

    for num in sorted(weight.keys()):
        print num, weight[num]
