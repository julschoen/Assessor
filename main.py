from temp_data_handler import Data4D
from trainer import Trainer
import argparse
import os


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--niters', type=int, default=1500, help='Number of training iterations')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--filterG', type=int, default=64, help='Number of filters G')
	parser.add_argument('--filterD', type=int, default=64, help='Number of filters D')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate imG')
	parser.add_argument('--data_path', type=str, default='../Data/4dct_clean',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
	parser.add_argument('--steps_per_img_log', type=int, default=50, help='Image Save Iterations')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--att', type=bool, default=True, help='Use Attention in BigGAN')
	parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')
	params = parser.parse_args()
	print(params)
	dataset_train = Data4D(path=os.path.join(params.data_path, 'train_pat.npz'))
	dataset_val = Data4D(path=os.path.join(params.data_path, 'test_pat.npz'))

	trainer = Trainer(dataset_train, dataset_val, params=params)
	trainer.train()

if __name__ == '__main__':
	main()
