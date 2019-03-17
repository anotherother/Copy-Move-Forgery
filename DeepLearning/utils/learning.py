from time import time
from tqdm import tqdm
import torch

def train_on_epoch(model, device, dataloader, loss_fn, optimizer):
	# Setup
	time_start = time()
	running_loss = 0
	n_steps = len(dataloader)

	# Train
	model.train()
	for (X, y) in tqdm(dataloader, total=n_steps):
		# Forward pass
		X, y = X.to(device), y.to(device)
		logits = model(X)
		loss = loss_fn(logits, y)
		iter_loss = loss.item()
		running_loss += iter_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_train = running_loss / n_steps
	time_exe = time() - time_start
	return loss_train, time_exe

def valid_on_epoch(model, device, dataloader, loss_fn):
	# Setup
	time_start = time()
	running_loss = 0
	n_steps = len(dataloader)

	# Validate
	model.eval()
	for (X, y) in tqdm(dataloader, total=n_steps):
		# Forward pass
		X, y = X.to(device), y.to(device)
		logits = model(X)
		loss = loss_fn(logits, y)
		running_loss += loss.item()

	loss_valid = running_loss / n_steps
	time_exe = time() - time_start
	return loss_valid, time_exe
