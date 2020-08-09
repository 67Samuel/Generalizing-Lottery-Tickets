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
		num_epochs = args.alexnet_epochs
		lr_anneal_epochs=[]
		for ms in args.milestone:
			lr_anneal_epochs.append(ms)
	else:
		raise ValueError(args.architecture + " architecture not supported")

	criterion = nn.CrossEntropyLoss().cuda()
	if args.optimizer == 'sgd':
		if args.architecture == "alexnet":
			optimizer = optim.SGD(model.parameters(), lr=args.alexnet_lr, momentum=0.9, weight_decay=0.004)
		else:
			optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
	else:
		raise ValueError(args.optimizer + " optimizer not supported")

	if (args.architecture == "vgg19") or (args.architecture == "alexnet"):
		model.apply(initialize_xavier_normal)
		
	wandb.init(entity="67Samuel", project='Varungohli Lottery Ticket', name=args.run_name, config={'batch size':args.batch_size, 'lr':optimizer.param_groups[0]['lr'], 'epochs':num_epochs})

	model.to(device)
	
	early_stopper = EarlyStopping(7)

	print(f"Started Training...")
	original_lr = optimizer.param_groups[0]['lr']
	for epoch in range(1, num_epochs+1):
		wandb.log({'epochs':epoch})
		
		if original_lr < (optimizer.param_groups[0]['lr'])*1000000:
			break
		
		if epoch in lr_anneal_epochs:
			optimizer.param_groups[0]['lr'] /= 10
			
		if (early_stopper.early_stop == True):
			if all(num > epoch for num in lr_anneal_epochs):
                		break
			else:
				optimizer.param_groups[0]['lr'] /= 10

		for batch_num, data in enumerate(dataloader, 0):
			inputs, labels = data[0].to(device), data[1].to(device)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			early_stopper(val_loss=loss, model=model)
			wandb.log({'train loss':loss.item()})
			loss.backward()
			optimizer.step()
			
		if epoch == 300:
			if args.optimizer == 'sgd':
				if args.architecture == "alexnet":
					optimizer.param_groups[0]['lr'] = args.alexnet_lr
				else:
					optimizer.param_groups[0]['lr'] = 0.1
			elif args.optimizer == 'adam':
				optimizer.param_groups[0]['lr'] = 0.0003
			else:
				print('cycle not supported for this optimizer type')

		if loss < 0.3:
			#if args.architecture == "resnet50":
			#	start_saving = 50
			#elif args.architecture == "vgg19":
			#	start_saving = 80
			#elif args.architecture == "alexnet":
			#	start_saving = 250
			if (epoch%(num_epochs/10) == 0):
				try:
					torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, args.model_saving_path + f"/{args.architecture}_{epoch}")
				except FileNotFoundError:
					print(args.model_saving_path + " path not found")
				
		wandb.log({'train lr':optimizer.param_groups[0]['lr']})
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


