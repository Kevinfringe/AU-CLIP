'''
	Option file for interpolation.
'''
from argparse import ArgumentParser


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', default='./results/cutomized_12th_with_auloss_simple_clip',type=str, help='Path to experiment output directory')
		self.parser.add_argument('--mapper_type', default='LevelsMapper', type=str, help='Which mapper to use')
		self.parser.add_argument('--no_coarse_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_medium_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_fine_mapper', default=False, action="store_true")
		self.parser.add_argument('--latents_train_path', default="../train_faces.pt", type=str, help="The latents for the training") # remove one "../" when customized training.
		self.parser.add_argument('--latents_test_path', default="../test_faces.pt", type=str, help="The latents for the validation") # remove one "../" when customized training.
		self.parser.add_argument('--train_dataset_size', default=5000, type=int, help="Will be used only if no latents are given")
		self.parser.add_argument('--test_dataset_size', default=1000, type=int, help="Will be used only if no latents are given")
		self.parser.add_argument('--work_in_stylespace', default=False, action='store_true', help="trains a mapper in S instead of W+")

		self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.5, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')

		self.parser.add_argument('--id_lambda', default=0.2, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--clip_lambda', default= 0.8, type=float, help='CLIP loss multiplier factor')
		self.parser.add_argument('--latent_l2_lambda', default=0.5, type=float, help='Latent L2 loss multiplier factor')

		self.parser.add_argument('--stylegan_weights', default='../stylegan2-ffhq-config-f.pt', type=str, help='Path to StyleGAN model weights')# remove one "../" when customized training.
		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		self.parser.add_argument('--ir_se50_weights', default='../model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")# remove one "../" when customized training.
		self.parser.add_argument('--checkpoint_path', default="./results/cutomized_15th_only_au_id_loss/checkpoints/iteration_50000.pt", type=str, help='Path to StyleCLIPModel model checkpoint')

		self.parser.add_argument('--max_steps', default=50000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=2000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=2000, type=int, help='Model checkpoint interval')

		self.parser.add_argument('--description', type=str, help='Driving text prompt')
		# added argument.
		self.parser.add_argument('--train_aus', default='./training/train_aus', type=str, help='Path to the action unit of train set.')
		self.parser.add_argument('--test_aus', default='./training/test_aus', type=str, help='Path to the action unit of test set.')
		self.parser.add_argument('--tensorboard_path', default='../../../tf_logs', type=str, help='Path to output tensorboard log for visualization.')# remove one "../" when customized training.
		self.parser.add_argument('--au_lambda', default=1, type=float, help='coefficient for action unit loss')
		self.parser.add_argument('--index_ls_path', default="./truncated_index.csv", type=str, help='coefficient for action unit loss')



	def parse(self):
		opts = self.parser.parse_args()
		return opts