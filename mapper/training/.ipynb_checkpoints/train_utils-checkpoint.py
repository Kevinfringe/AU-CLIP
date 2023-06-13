# The resulting STYLESPACE_DIMENSIONS list contains 25 elements, each of which represents the number of channels in a convolutional layer of the style-based synthesis network. The first 15 elements have a value of 512, the next three elements have a value of 256, the next three elements have a value of 128, the next three elements have a value of 64, and the final two elements have a value of 32.
STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals


def convert_s_tensor_to_list(batch):
	s_list = []
	for i in range(len(STYLESPACE_DIMENSIONS)):
		s_list.append(batch[:, :, 512 * i: 512 * i + STYLESPACE_DIMENSIONS[i]])
	return s_list
