#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile
import oversegment


cProfile.run("oversegment.main()", "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
