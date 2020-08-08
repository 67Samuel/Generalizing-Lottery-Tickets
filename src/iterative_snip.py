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

def prune_iteratively(model, run_name, batch_size, img_size, dataloader, architecture, optimizer_type, device, models_path, random, is_equal_classes, reinit, alexnet_epochs, alexnet_lr, cycle_epoch=10000000):
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
	is_equal_classes : Boolean to indicate is source and target dataset have equal number of classes
	Returns
	--------
	None
	"""
	if architecture == "vgg19":
		num_epochs = 160
		lr_anneal_epochs = [80, 120]
	elif architecture == "resnet50":
		num_epochs = 90
		lr_anneal_epochs = [50, 65, 80]
	elif architecture == "alexnet":
		num_epochs = alexnet_epochs
		lr_anneal_epochs=[]
		for ms in args.milestone:
			lr_anneal_epochs.append(ms)
	elif architecture == "test_resnet50":
		num_epochs = 3
		lr_anneal_epochs = []
	else:
		raise ValueError(architecture + " architecture not supported")

	criterion = nn.CrossEntropyLoss().cuda()

	if optimizer_type == 'sgd':
		if architecture == "alexnet":
			wandb.init(entity="67Samuel", project='Varungohli SNIP', name=run_name, config={'batch size':args.batch_size, 'lr':alexnet_lr, 'epochs':num_epochs})
		else:
			wandb.init(entity="67Samuel", project='Varungohli SNIP', name=run_name, config={'batch size':args.batch_size, 'lr':0.1, 'epochs':num_epochs})
	elif optimizer_type == 'adam':
		wandb.init(entity="67Samuel", project='Varungohli SNIP', name=run_name, config={'batch size':args.batch_size, 'lr':0.0003, 'epochs':num_epochs})
	else:
		wandb.init(entity="67Samuel", project='Varungohli SNIP', name=run_name, config={'batch size':args.batch_size, 'epochs':num_epochs})
	print("Iterative Pruning started")
	for pruning_iter in range(0,31):
		wandb.log({'prune iteration':pruning_iter})
		print(f"Running pruning iteration {pruning_iter}")
		if optimizer_type == 'sgd':
			if architecture == "alexnet":
				optimizer = optim.SGD(model.parameters(), lr=alexnet_lr, momentum=0.9, weight_decay=0.004)
			else:
				optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
		elif optimizer_type == 'adam':
			optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
		else:
			raise ValueError(optimizer_type + " optimizer not supported")

		if pruning_iter != 0:
			cpt = torch.load(models_path + f"/{pruning_iter-1}_{num_epochs}")
			model.load_state_dict(cpt['model_state_dict'])
			
		model.to(device)
			
		snip_factor = round((100*(0.8**(pruning_iter+1)))/100, 5)
		print(f"Pruning 20% of latest model weights with SNIP, snip factor: {snip_factor}...")
		keep_masks = SNIP(model, snip_factor, dataloader, device, img_size=img_size)
		# Reinit weights
		if reinit:
			model.apply(weight_reset)
		# Apply mask
		apply_prune_mask(model, keep_masks)

		for epoch in range(1, num_epochs+1):
			wandb.log({'epochs':epoch})
			if epoch in lr_anneal_epochs:
				optimizer.param_groups[0]['lr'] /= 10

			for batch_num, data in enumerate(dataloader, 0):
				inputs, labels = data[0].to(device), data[1].to(device)
				optimizer.zero_grad()

				outputs = model(inputs)
				loss = criterion(outputs, labels)
				wandb.log({'prune loss':loss})
				loss.backward()
				optimizer.step()
				
			if epoch == cycle_epoch:
				if optimizer_type == 'sgd':
					if architecture == "alexnet":
						optimizer.param_groups[0]['lr'] = alexnet_lr
					else:
						optimizer.param_groups[0]['lr'] = 0.1
				elif optimizer_type == 'adam':
					optimizer.param_groups[0]['lr'] = 0.0003
				else:
					print('cycle not supported for this optimizer type')
				
			wandb.log({'train lr':optimizer.param_groups[0]['lr']})

			if epoch == num_epochs:
				print(f'saving checkpoint to {models_path}/{str(pruning_iter)}_{str(num_epochs)}...')
				torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict() },models_path + "/"+ str(pruning_iter) + "_" + str(num_epochs))
	print("Finished Iterative Pruning")
	print("Please delete the saved model state dicts from iter 16 and below to free up space")


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
	if args.architecture == "test_resnet50":
		model = load_model("resnet50", num_classes_target)
	else:
		model = load_model(args.architecture, num_classes_target)
  
  #Get img size
	if args.target_dataset in ['cifar10', 'cifar100', 'svhn', 'cifar10a', 'cifar10b']:
		img_size = 32
	elif args.target_dataset == 'fashionmnist':
		img_size = 28
	else:
		raise ValueError(args.target_dataset + " dataset not supported")

	if num_classes_source == num_classes_target:
		prune_iteratively(model, args.run_name, args.batch_size, img_size, dataloader, args.architecture, args.optimizer, device, args.model_saving_path, args.random, True, args.reinit, args.alexnet_epochs, args.alexnet_lr, args.cycle_epoch)
	else:
		prune_iteratively(model, args.run_name, args.batch_size, img_size, dataloader, args.architecture, args.optimizer, device, args.model_saving_path, args.random, False, args.reinit, args.alexnet_epochs, args.alexnet_lr, args.cycle_epoch)
