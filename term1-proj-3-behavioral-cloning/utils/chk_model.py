# -*- coding: utf-8 -*-

import os
import argparse
import imp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='show model summary')
    parser.add_argument('--module_path', type=str, default='model.py', help='file name to python file that contain the definition python.')
    args = parser.parse_args()
    module_path = args.module_path
    module_name = module_path.strip().split('/')[-1]
    if os.path.isfile(module_path):
        with open(module_path, 'rb') as f:
            module = imp.load_module('get_model', f, module_name, ('.py', 'rb', imp.PY_SOURCE))
        model = module.get_model()
        model.summary()
    else:
        print('{} does not exist in current directory'.format(module_path))
