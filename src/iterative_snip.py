from utils import *
import torch
import random
import torchvision
import torch.optim as optim
import numpy as np
import wandb
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

def initialize_xavier_normal(layer):
	"""
	Function to initialize a layer by picking weights from a xavier normal distribution
	Arguments
	---------
	layer : The layer of the neural network
	Returns
	-------
	None
	"""
	if type(layer) == nn.Conv2d:
		torch.nn.init.xavier_normal_(layer.weight)
		layer.bias.data.fill_(0)

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device, img_size=None, num_channels=3):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    if type(img_size) == int:
        inputs = inputs.view(-1,num_channels,img_size,img_size).float().requires_grad_()
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
    
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
    
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)

def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask
            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask)) #hook masks onto respective weights
	
def weight_reset(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		m.reset_parameters()

def prune_iteratively(model, args, img_size, dataloader, device):
	"""
	Performs iterative pruning
	Arguments
	---------
	model : the PyTorch neural network model to be trained
	dataloader : PyTorch dataloader for loading the dataset
	args.architecture : The neural network architecture (VGG19 or ResNet50)
	args.optimizer : The optimizer to use for training (SGD / Adam)
	device : Device(GPU/CPU) on which to perform computation
	args.model_saving_path: Path to directory where trained model/checkpoints will be saved
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
	
	if args.wandb:
		# run wandb init
		if args.optimizer == 'sgd':
			lr=0.01
		elif args.optimizer == 'adam':
			lr=0.0003
		wandb.init(entity=args.entity, project=args.project, name=args.run_name, config={'batch size':args.batch_size, 'lr':lr, 'epochs':num_epochs})
	
	print("Iterative Pruning started")
	for pruning_iter in range(0,31):
		if args.wandb:
			# log each iteration to wandb
			wandb.log({'prune iteration':pruning_iter})
		print(f"Running pruning iteration {pruning_iter}")
		if args.optimizer == 'sgd':
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
		elif args.optimizer == 'adam':
			optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
		else:
			raise ValueError(args.optimizer + " optimizer not supported")
			
		if (args.architecture == "vgg19") or (args.architecture == "alexnet"):
			model.apply(initialize_xavier_normal)

		if pruning_iter != 0:
			cpt = torch.load(args.model_saving_path + f"/{pruning_iter-1}_{num_epochs}")
			model.load_state_dict(cpt['model_state_dict'])
			
		model.to(device)
			
		# snip percentage to increase by 20% for each iteration
		snip_factor = round((100*(0.8**(pruning_iter+1)))/100, 5)
		print(f"Pruning 20% of latest model weights with SNIP, snip factor: {snip_factor}...")
		keep_masks = SNIP(model, snip_factor, dataloader, device, img_size=img_size)
		# Reinitialise weights
		if args.reinit:
			model.apply(weight_reset)
		# Apply mask
		apply_prune_mask(model, keep_masks)

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
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				if args.wandb:
					# log loss at each epoch
					wandb.log({'prune loss':loss.item()})
				loss.backward()
				optimizer.step()
			if args.wandb:
				# log lr at each epoch
				wandb.log({'train lr':optimizer.param_groups[0]['lr']})
			
			if (epoch == num_epochs):
				# save model at the end of each iteration, file looks like 1_500, 2_500 etc
				print(f'saving checkpoint to {args.model_saving_path}/{str(pruning_iter)}_{str(num_epochs)}...')
				torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict() },args.model_saving_path + "/"+ str(pruning_iter) + "_" + str(num_epochs))
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


	#Checks number of classes for appropriate linear layer at end of model
	if args.source_dataset in ['cifar10', 'svhn', 'fashionmnist']:
		num_classes_source = 10
	elif args.source_dataset in ['cifar100']:
		num_classes_source = 100
	else:
		raise ValueError(args.source_dataset + " as a source dataset is not supported")

	#Loads dataset
	dataloader = load_dataset(args.target_dataset, args.batch_size, True)

	#Loads model
	model = load_model(args.architecture, num_classes_target)
 
	# Get image size depending on dataset to use in SNIP function
	if args.target_dataset in ['cifar10', 'cifar100', 'svhn', 'cifar10a', 'cifar10b']:
		img_size = 32
	elif args.target_dataset == 'fashionmnist':
		img_size = 28
	else:
		raise ValueError(args.target_dataset + " dataset not supported")

	prune_iteratively(model, args, img_size, dataloader, device)
