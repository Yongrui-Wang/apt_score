import torch
import time
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataset import APTDataset

from utils import plot_image, plot_curve, one_hot

__author__ = "Yongrui Wang"
__license__ = "MIT"


dataset = APTDataset()
batch_size = 512
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# step1. load dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class Net(nn.Module):
    def __init__(self, hidden_size=20):
        super(Net, self).__init__()
        self.concat_size = 34*47 + 340
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=19, kernel_size=6),
            nn.Tanh(),
            nn.Conv1d(in_channels=19, out_channels=16, kernel_size=7),
            nn.Tanh(),
            nn.Conv1d(in_channels=16, out_channels=34, kernel_size=11)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.concat_size, 658),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(658, 160),
            nn.Tanh(),
            nn.Dropout(0.005),
            nn.Linear(160, 123),
            nn.Tanh(),
            nn.Dropout(0.256),
            nn.Linear(123, 1)
        )

    def forward(self, one_hot, kmer):
        one_hot = one_hot.reshape(one_hot.size(0), one_hot.size(2), one_hot.size(1))
        one_hot = self.conv_layers(one_hot)
        one_hot = one_hot.flatten(start_dim=1)
        x = torch.cat((one_hot, kmer), dim=-1)
        x = self.linear_layers(x)

        return x




# net = Net()
# [w1, b1, w2, b2, w3, b3]
device = torch.device('cuda')
model = Net().to(device)
# x = x.to(device)
# y = y.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.000003,  weight_decay=1e-5)
criterion = nn.MSELoss(reduction='mean')

writer = SummaryWriter()
step = 0
train_loss = []
eval_preds = []
eval_y = []

for epoch in tqdm(range(40)):
    model.train()
    loss_record = []
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        one_hot= x['one_hot']
        kmer = x['kmer']
        one_hot = one_hot.reshape(one_hot.size(0), 68, 4)
        one_hot, kmer , y= one_hot.to(device), kmer.to(device), y.to(device)
        # => [b, 10]
        out = model(one_hot, kmer)
        # loss = mse(out, y_onehot)

        loss = criterion(out, y)

        loss.backward()
        # w' = w - lr*grad
        optimizer.step()

        step += 1
        loss_record.append(loss.detach().item())
        train_loss.append(loss.item())
    mean_train_loss = sum(loss_record)/len(loss_record)
    writer.add_scalar('Loss/train', mean_train_loss, step)


    model.eval()
    loss_record = []
    for x, y in test_loader:
        one_hot= x['one_hot']
        kmer = x['kmer']
        one_hot = one_hot.reshape(one_hot.size(0), 68, 4)
        one_hot, kmer , y= one_hot.to(device), kmer.to(device), y.to(device)
        with torch.no_grad():
            pred = model(one_hot, kmer)
            loss = criterion(pred, y)

        loss_record.append(loss.item())
    mean_valid_loss = sum(loss_record)/len(loss_record)
    print(print(f'Epoch [{epoch+1}/{20}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}'))
    writer.add_scalar('Loss/valid', mean_valid_loss, step)

plot_curve(eval_preds)
# we get optimal [w1, b1, w2, b2, w3, b3]


# total_correct = 0
# for x,y in test_loader:
#     x = x.view(x.size(0), 28*28)
#     x = Variable(x).cuda()
#     y = Variable(y).cuda()
#     out = model(x)
#     # out: [b, 10] => pred: [b]
#     out = Variable(out).cuda()
#     pred = out.argmax(dim=1)
#     correct = pred.eq(y).sum().float().item()
#     total_correct += correct
#
# total_num = len(test_loader.dataset)
# acc = total_correct / total_num
# print('test acc:', acc)
#
# x, y = next(iter(test_loader))
# model = model.cpu()
# out = model(x.view(x.size(0), 28*28))
# pred = out.argmax(dim=1)
# plot_image(x, pred, 'test')
