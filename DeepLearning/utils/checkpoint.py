import os
import torch
from scipy.io import savemat

class CheckPoint(object):
	def __init__(self, model, optimizer, loss_fn, savedir, improved_delta=0.01, last_best_loss=inf):
		super(CheckPoint, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.savedir = savedir
		self.improved_delta = improved_delta
		self.best_loss = last_best_loss
		if not os.path.exists(savedir):
			os.makedirs(savedir)

	def backup(self, loss_train, loss_valid, acc_train, acc_valid, metrics, epoch):
		if (self.best_loss-loss_valid)>=self.improved_delta:
			checkpoint = {
				"epoch": epoch,
				"model": self.model.state_dict(),
				"optimizer": self.optimizer.state_dict(),
				"loss_fn": self.loss_fn,

				"loss_train": loss_train, "loss_valid": loss_valid,
				"acc_train": acc_train, "acc_valid": acc_valid,
			}
			fname_checkpoint = "checkpoint-epoch%d-loss%f.ckpt" % (epoch, loss_valid)
			file_checkpoint = os.path.join(self.savedir, fname_checkpoint)
			torch.save(checkpoint, file_checkpoint)

			file_metrics = os.path.join(self.savedir, "metrics.mat")
			savemat(file_metrics, metrics)

			print("Loss improved from %f to %f" % (self.best_loss, loss_valid))
			print("Checkpoint saved in", file_checkpoint)
			print("Metrics saved in", file_metrics)
			self.best_loss = loss_valid

		else:
			file_metrics = os.path.join(self.savedir, "metrics.mat")
			savemat(file_metrics, metrics)
			print("Metrics saved in", file_metrics)
			print("Not improved enough from %f" % (self.best_loss))

	def reload(self, last_best_loss, improved_delta):
		self.last_best_loss = last_best_loss
		self.improved_delta = improved_delta


#------------------------------------------------------------------------------
#	Early Stopping
#------------------------------------------------------------------------------
class EarlyStopping(object):
	def __init__(self, not_improved_thres=1,
					improved_delta=0.01,
					last_best_loss=inf):
		super(EarlyStopping, self).__init__()
		self.not_improved_thres = not_improved_thres
		self.improved_delta = improved_delta
		self.not_improved = 0
		self.best_loss = last_best_loss


	def check(self, loss):
		if (self.best_loss-loss)>=self.improved_delta:
			self.not_improved = 0
			self.best_loss = loss
		else:
			self.not_improved += 1
			print("Not improved enough quantity: %d times" % (self.not_improved))
			if self.not_improved==self.not_improved_thres:
				print("Early stopping")
				return True
		return False


	def reload(self, last_best_loss, not_improved_thres, improved_delta):
		self.not_improved = 0
		self.last_best_loss = last_best_loss
		self.not_improved_thres = not_improved_thres
		self.improved_delta = improved_delta
