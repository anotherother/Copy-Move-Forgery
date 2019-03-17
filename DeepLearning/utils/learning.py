from time import time
from tqdm import tqdm
import torch

def train_on_epoch(model, device, dataloader, loss_fn, optimizer, epoch):
	# Setup
	time_start = time()
	running_loss = 0
	running_corrects = 0
	n_steps = len(dataloader)
	n_samples = len(dataloader.dataset)

	# Train
	model.train()
	print("Train on epoch %d" % (epoch))
	for (X, y) in tqdm(dataloader, total=n_steps):
		# Forward pass
		X, y = X.to(device), y.to(device)
		logits = model(X)
		loss = loss_fn(logits, y)
		_, preds = torch.max(logits.data, 1)
		iter_loss = loss.item()
		running_loss += iter_loss
		iter_correct = torch.sum(preds==y).item()
		running_corrects += iter_correct

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_train = running_loss / n_steps
	acc_train = running_corrects / n_samples
	time_exe = time() - time_start
	return loss_train, acc_train, time_exe

def valid_on_epoch(model, device, dataloader, loss_fn, epoch):
	# Setup
	time_start = time()
	running_loss = 0
	running_corrects = 0
	n_steps = len(dataloader)
	n_samples = len(dataloader.dataset)

	# Validate
	model.eval()
	print("Validate on epoch %d" % (epoch))
	for (X, y) in tqdm(dataloader, total=n_steps):
		# Forward pass
		X, y = X.to(device), y.to(device)
		logits = model(X)
		loss = loss_fn(logits, y)
		_, preds = torch.max(logits.data, 1)
		running_loss += loss.item()
		running_corrects += torch.sum(preds==y).item()

	loss_valid = running_loss / n_steps
	acc_valid = running_corrects / n_samples
	time_exe = time() - time_start
	return loss_valid, acc_valid, time_exe
