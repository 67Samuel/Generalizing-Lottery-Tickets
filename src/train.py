from utils import *
import torch
import random
import torchvision
import torch.optim as optim
import wandb
import numpy as np


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

def train(model, args, dataloader, device):
	"""
	Function to train the network 

	Arguments
	---------
	model : the PyTorch neural network model to be trained
	dataloader : PyTorch dataloader for loading the dataset
	architecture : The neural network architecture (VGG19 or ResNet50)
	optimizer_type : The optimizer to use for training (SGD / Adam)
	device : Device(GPU/CPU) on which to perform computation
	model_path: Path to directory where trained model/checkpoints will be saved

	Returns
	-------
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
	
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.004)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
	else:
		raise ValueError(args.optimizer + " optimizer not supported")

	if (args.architecture == "vgg19") or (args.architecture == "alexnet"):
		model.apply(initialize_xavier_normal)
		
	if args.wandb:
		wandb.init(entity=args.entity, project=args.project, name=args.run_name, config={'batch size':args.batch_size, 'lr':optimizer.param_groups[0]['lr'], 'epochs':num_epochs})

	model.to(device)
	
	# initialise early stopper to stop model when loss does not increase after a certain patience
	early_stopper = EarlyStopping(args.esp)

	print(f"Started Training...")
	original_lr = optimizer.param_groups[0]['lr']
	for epoch in range(1, num_epochs+1):
		if args.wandb:
			# log each epoch
			wandb.log({'epochs':epoch})
		
		# if lr becomes too low due to early stopper patience running out too many times, stop training
		if original_lr > (optimizer.param_groups[0]['lr'])*(10*len(lr_anneal_epochs)+1000000):
			print(f"lr is too low at {optimizer.param_groups[0]['lr']}")
			break
		
		if epoch in lr_anneal_epochs:
			# decrease lr after previously set epochs
			optimizer.param_groups[0]['lr'] /= 10
		
		#if (early_stopper.early_stop == True):
			# if there are still more lr milestones that haven't come, derease lr and continue. If all lr milestones have been completed, stop the training
			#if all(num < epoch for num in lr_anneal_epochs):
				#print(f"epoch {epoch} > {lr_anneal_epochs[-1]} and early stopping is true. Stopping training...")
				#break
			#else:
				#optimizer.param_groups[0]['lr'] /= 2

		for batch_num, data in enumerate(dataloader, 0):
			inputs, labels = data[0].to(device), data[1].to(device)
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			# call early stopper
			if args.wandb:
				# log loss after each batch
				wandb.log({'train loss':loss.item()})
			loss.backward()
			optimizer.step()
		
		# save model only if loss is past 0.3 to save memory
		if (epoch == num_epochs):
			try:
				# saved file looks like alexnet_500 etc
				torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, args.model_saving_path + f"/{args.architecture}_{epoch}")
			except FileNotFoundError:
				print(args.model_saving_path + " path not found")
		if args.wandb:
			#log lr at each epoch
			wandb.log({'train lr':optimizer.param_groups[0]['lr']})
		early_stopper(val_loss=loss, model=model)
		print(f"Epoch {epoch} : Loss = {loss.item()}")
	print("Finished Training!")

if __name__ == '__main__':
	#Parsers the command line arguments
	parser = args_parser_train()
	args = parser.parse_args()

	#Sets random seed
	random.seed(args.seed)

	#Uses GPU is available
	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device.')

	#Loads dataset
	dataloader = load_dataset(args.dataset, args.batch_size, True)

	#Checks number of classes to aa appropriate linear layer at end of model
	if args.dataset in ['cifar10', 'fashionmnist', 'svhn']:
		num_classes = 10
	elif args.dataset in ['cifar100']:
		num_classes = 100
	else:
		raise ValueError(args.dataset + " dataset not supported")

	#Loads model
	model = load_model(args.architecture, num_classes)
	if args.load_model != None:
		try:
			cpt = torch.load(str(args.load_model))
		except Exception as e:
			print(e)
			cpt = torch.load("C:/Users/user/OneDrive - Singapore University of Technology and Design/Desktop/new_ver/src/" + args.load_model)
		model.load_state_dict(cpt['model_state_dict'])

	train(model, args, dataloader, device)


