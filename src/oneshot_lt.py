from utils import *
import torch
import random
import torchvision
import torch.optim as optim
import numpy as np
import wandb


def permute_masks(old_masks):
	""" 
	Function to randomly permute the mask in a global manner.
	Arguments
	---------
	old_masks: List containing all the layer wise mask of the neural network, mandatory. No default.
	seed: Integer containing the random seed to use for reproducibility. Default is 0
	Returns
	-------
	new_masks: List containing all the masks permuted globally
	"""

	layer_wise_flatten = []                      # maintain the layerwise flattened tensor
	for i in range(len(old_masks)):
		layer_wise_flatten.append(old_masks[i].flatten())

	global_flatten = []
	for i in range(len(layer_wise_flatten)):
		if len(global_flatten) == 0:
			global_flatten.append(layer_wise_flatten[i].cpu())
		else:
			global_flatten[-1] = np.append(global_flatten[-1], layer_wise_flatten[i].cpu())
	permuted_mask = np.random.permutation(global_flatten[-1])

	new_masks = []
	idx1 = 0
	idx2 = 0
	for i in range(len(old_masks)):
		till_idx = old_masks[i].numel()
		idx2 = idx2 + till_idx
		new_masks.append(permuted_mask[idx1:idx2].reshape(old_masks[i].shape))
		idx1 = idx2

	# Convert to tensor
	for i in range(len(new_masks)):
		new_masks[i] = torch.tensor(new_masks[i])

	return new_masks


def prune(model, args, dataloader, device):
	"""
	Performs iterative pruning
	Arguments
	---------
	model : the PyTorch neural network model to be trained
	dataloader : PyTorch dataloader for loading the dataset
	architecture : The neural network architecture (VGG19 or ResNet50)
	optimizer_type : The optimizer to use for training (SGD / Adam)
	device : Device(GPU/CPU) on which to perform computation
	models_path: Path to directory where trained model/checkpoints will be saved
	init_path : Path to winning ticket initialization model
	random    : Boolean which when True perform pruning for random ticket
	Returns
	--------
	None
	"""
	if args.architecture == "vgg19":
		num_epochs = 160
		lr_anneal_epochs = [80, 120]
	elif args.architecture == "resnet50":
		num_epochs = 90
		lr_anneal_epochs = [50, 65, 80]
	elif args.architecture == "alexnet":
		num_epochs = 500
		lr_anneal_epochs = [450, 470, 480, 490]
	else:
		raise ValueError(args.architecture + " architecture not supported")

	criterion = nn.CrossEntropyLoss().cuda()

	weight_fractions = get_weight_fractions()
	
	if args.wandb:
		# run wandb init
		if args.optimizer == 'sgd':
			lr=0.01
		elif args.optimizer == 'adam':
			lr=0.0003
		wandb.init(entity=args.entity, project=args.project, name=args.run_name, config={'batch size':args.batch_size, 'lr':lr, 'epochs':num_epochs})
	

	cpt = torch.load(args.models_path + f"/{pruning_iter-1}_{num_epochs}")
	model.load_state_dict(cpt['model_state_dict'])

	# apply mask
	masks = []
	flat_model_weights = np.array([])
	for name, params in model.named_parameters():
		if "weight" in name:
			layer_weights = params.data.cpu().numpy()
			flat_model_weights = np.concatenate((flat_model_weights, layer_weights.flatten()))
	threshold = np.percentile(abs(flat_model_weights), args.snip)

	zeros = 0
	total = 0
	for name, params in model.named_parameters():
		if "weight" in name:
			weight_copy = params.data.abs().clone()
			mask = weight_copy.gt(threshold).float()
			zeros += mask.numel() - mask.nonzero().size(0)
			total += mask.numel()
			masks.append(mask)
			if args.random != 'false':
				masks = permute_masks(masks)
	print(f"Fraction of weights pruned = {zeros}/{total} = {zeros/total}")  

	# set optimizer
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.004)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
	else:
		raise ValueError(args.optimizer_type + " optimizer not supported")

	model.to(device)

	for epoch in range(1, num_epochs+1):
		if args.wandb:
			# log each epoch
			wandb.log({'epochs':epoch})
		if epoch in lr_anneal_epochs:
			# decrease lr at previously specified epochs
			optimizer.param_groups[0]['lr'] /= 10

		for batch_num, data in enumerate(dataloader, 0):
			inputs, labels = data[0].to(device), data[1].to(device)
			optimizer.zero_grad()

			if pruning_iter != 0:
				layer_index = 0
				for name, params in model.named_parameters():
					if "weight" in name:
						params.data.mul_(masks[layer_index].to(device))
						layer_index += 1

			outputs = model(inputs)
			loss = criterion(outputs, labels)
			if args.wandb:
				# log loss at each epoch
				wandb.log({'prune loss':loss})
			loss.backward()
			optimizer.step()
			
		if args.wandb:
			# log lr at each epoch
			wandb.log({'train lr':optimizer.param_groups[0]['lr']})
		if (epoch == num_epochs):
			print('saving model...')
			if pruning_iter != 0:
				layer_index = 0
				for name, params in model.named_parameters():
					if "weight" in name:
						params.data.mul_(masks[layer_index].to(device))
						layer_index += 1
			torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict() },args.model_saving_path + "/" + str(epoch))
	print("Finished Iterative Pruning")

if __name__ == '__main__':
	#Parsers the command line arguments
	parser = args_parser_iterprune()
	args = parser.parse_args()

	#Sets random seed
	random.seed(args.seed)

	#Uses GPU is available
	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device.')


	#Checks number of classes to aa appropriate linear layer at end of model
	if args.source_dataset in ['cifar10', 'svhn', 'fashionmnist']:
		num_classes_source = 10
	elif args.source_dataset in ['cifar100']:
		num_classes_source = 100
	else:
		raise ValueError(args.source_dataset + " as a source dataset is not supported")

	if args.target_dataset in ['cifar10', 'svhn', 'fashionmnist']:
		num_classes_target = 10
	elif args.target_dataset in ['cifar100']:
		num_classes_target = 100
	else:
		raise ValueError(args.target_dataset + " as a target dataset is not supported")

	#Loads dataset
	dataloader = load_dataset(args.target_dataset, args.batch_size, True)

	#Loads model
	model = load_model(args.architecture, num_classes_target)

	# for trasfer learning
	if num_classes_source == num_classes_target:
		prune_iteratively(model, args, dataloader, device, True)
	else:
		prune_iteratively(model, args, dataloader, device, False)
