import os
import cv2
import time
import matplotlib.pyplot as plt


class Logger:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.plot_dir = os.path.join(log_dir, 'plots')
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.media_dir = os.path.join(log_dir, 'media')
        if not os.path.exists(self.media_dir):
            os.mkdir(self.media_dir)

        self.log_file_path = self.log_dir + '/logs.txt'
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write('Logs date and time: '+time.strftime("%d-%m-%Y %H:%M:%S")+'\n\n')

        self.train_loss = []
        self.psnr = []

    def log(self, tag, **kwargs):

        self.log_file = open(self.log_file_path, 'a')

        if tag == 'args':
            self.log_file.write('Training Args:\n')
            for k, v in kwargs.items():
                self.log_file.write(str(k)+': '+str(v)+'\n')
            self.log_file.write('#########################################################\n\n')
            self.log_file.write(f'Starting Training... \n')

        elif tag == 'msg':
            self.log_file.write('#########################################################\n')
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Iteration: {kwargs["iter"]} \t {kwargs["msg"]}\n')
            self.log_file.write('#########################################################\n')

        elif tag == 'train':
            self.train_loss.append([kwargs['loss']])
            self.psnr.append([kwargs['psnr']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Iteration: {kwargs["iter"]} \t Train Loss: {kwargs["loss"]} \t PSNR: {kwargs["psnr"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'val':
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Iteration: {kwargs["iter"]} \t Val Loss: {kwargs["loss"]} \t PSNR: {kwargs["psnr"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'model':
            self.log_file.write('#########################################################\n')
            self.log_file.write(f'Saving model... Train Loss: {kwargs["loss"]}\n')
            self.log_file.write('#########################################################\n')

        elif tag == 'model_loss':
            self.log_file.write('#########################################################\n')
            self.log_file.write(f'Saving best model... Test Loss: {kwargs["loss"]}\n')
            self.log_file.write('#########################################################\n')

        elif tag == 'plot':
            self.plot(self.train_loss, name='Train Loss', path=self.plot_dir)
            self.plot(self.psnr, name='PSNR', path=self.plot_dir)

        self.log_file.close()

    def draw(self, epoch, img):

        cv2.imwrite(self.media_dir+'/'+str(epoch)+'.png', img)

    def plot(self, data, name, path):

        plt.plot(data)
        plt.xlabel('Iterations (x1000)')
        plt.ylabel(name)
        plt.title(name+' vs. Iterations (x1000)')
        plt.savefig(os.path.join(path, name+'.png'), dpi=600 ,bbox_inches='tight')
        plt.close()

    def plot_both(self, data1, data2, name, path):

        plt.plot(data1, label='Train')
        plt.plot(data2, label='Val')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.title(name+' vs. Epochs')
        plt.legend()
        plt.savefig(os.path.join(path, name+'.png'), dpi=600 ,bbox_inches='tight')
        plt.close()