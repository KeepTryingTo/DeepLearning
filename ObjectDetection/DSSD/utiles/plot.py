"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/2/7 21:27
"""

import matplotlib.pyplot as plt

def plot_loss(loss_train,loss_val):
    x = [i for i in range(len(loss_train))]
    fig,ax = plt.subplots()
    ax.plot(x,loss_train,'b^',label = 'loss_cls')
    ax.plot(x,loss_val,'g^',label = 'loss_reg')
    ax.set_xlabel(xlabel='epoch')
    ax.set_ylabel(ylabel='lossAndlr')
    ax.set_title('loss and lr')
    ax.legend()
    plt.savefig('runs/result_loss.png')
    plt.show()