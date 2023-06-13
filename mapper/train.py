"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint



sys.path.append(".")
sys.path.append("../")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/models")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/utils")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/Action-Units-Heatmaps-master")


from mapper.options.train_options import TrainOptions
from mapper.solver import Solver


def main(opts):
	# if os.path.exists(opts.exp_dir):
	# 	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	print("entered here")
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	solver = Solver(opts)
	solver.train()


if __name__ == '__main__':
	print("entered main branch.")
	opts = TrainOptions().parse()
	main(opts)
