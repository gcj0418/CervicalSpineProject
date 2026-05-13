#!/usr/bin/env python3
import sys
import os

# Ensure src and VLD repo are in path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'src'))
sys.path.insert(0, os.path.join(base_dir, '..', 'Vertebra-Landmark-Detection'))

from src.gui import main

if __name__ == '__main__':
    main()
