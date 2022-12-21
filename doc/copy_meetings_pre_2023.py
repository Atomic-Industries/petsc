#!/usr/bin/env python
'''Copies over old meeting webpages from community/meetings/pre-2023 to appropriate location in Sphinx generated directories'''

import os
import errno
import subprocess
import shutil
import argparse

def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise

def copy(outdir):
    bdir = os.path.join('community','meetings','pre-2023')
    for source_dir in os.listdir(bdir):
      if os.path.isdir(os.path.join(bdir,source_dir)) and not source_dir == '.git':
        target_dir = os.path.join(outdir, source_dir)
        source_dir = os.path.join(bdir,source_dir)
        print('Copying directory %s to %s' % (source_dir, target_dir))
        _mkdir_p(target_dir)
        for file in os.listdir(source_dir):
            source = os.path.join(source_dir, file)
            target = os.path.join(target_dir, file)
            if os.path.isdir(source):
                if os.path.isdir(target):
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                shutil.copy(source, target)

