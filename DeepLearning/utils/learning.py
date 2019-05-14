from tqdm import tqdm

def train_step(model, dataloader, loss_fn, optimizer):

	running_loss = 0
	n_steps = len(dataloader)

	model.train()
	for (image, mask) in tqdm(dataloader, total=n_steps):

		image, mask = image.cuda(), mask.cuda()
		logits = model(image)
		loss = loss_fn(logits.view(-1), mask.view(-1))
		iter_loss = loss.item()
		running_loss += iter_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_train = running_loss / n_steps

	return loss_train

def valid_step(model, dataloader, loss_fn):

	running_loss = 0
	n_steps = len(dataloader)

	model.eval()
	for (image, mask) in tqdm(dataloader, total=n_steps):

		image, mask = image.cuda(), mask.cuda()
		logits = model(image)
		loss = loss_fn(logits.view(-1), mask.view(-1))
		running_loss += loss.item()

	loss_valid = running_loss / n_steps
	return loss_valid
