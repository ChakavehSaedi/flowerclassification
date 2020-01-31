import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

class net_handler():
    def __init__(self, net_param, out_dir, pretrained = False):
        super(net_handler, self, ).__init__()

        self.out_dir = out_dir

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.cuda = True
        else:
            self.cuda = False

        self.net = neural_network(net_param, self.cuda)

        if pretrained:
            self.net.load_state_dict(torch.load(self.out_dir+"net.h5"))

        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0005, betas=(0.9, 0.999))

    def train_net(self, epochs, train_samples, val_samples, verbos, log_file):
        # Trainign the network
        epoch_loss = [0 for _ in range(epochs)]
        epoch_acc = [0 for _ in range(epochs)]

        val_loss = [0 for _ in range(epochs)]
        val_acc = [0 for _ in range(epochs)]

        for epoch in range(epochs):
            epoch_st = datetime.now()
            running_loss = 0.0

            # train
            for i, batch in enumerate(train_samples, 0):
                if self.cuda:
                    inputs, labels = batch.to('cuda')
                else:
                    inputs, labels = batch

                self.optimizer.zero_grad()
                prediction = self.net(inputs)
                batch_loss = self.criterion(prediction, labels)
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss[epoch] += batch_loss.item()

                max_indx = torch.max(prediction,1)[1].tolist()
                batch_acc = [1 if max_indx[j] == labels[j] else 0 for j in range(labels.shape[0])].count(1)
                batch_acc /= labels.shape[0]

                epoch_acc[epoch] += batch_acc

                # print statistics
                if verbos:
                    running_loss += batch_loss.item()
                    if i % 1000 == 0:
                        loss_1000 = running_loss / 1000
                        print('   -  Epoch %d, Batch %5d loss: %.3f' % (epoch + 1, i + 1, loss_1000))
                        with open(log_file, 'a') as log:
                            log.write("   - Epoch %d, Batch %5d loss: %.3f\n" % (epoch + 1, i + 1, loss_1000))
                        running_loss = 0.0

            epoch_loss[epoch] /= i
            epoch_acc[epoch] /= i

            val_acc[epoch], val_loss[epoch] = self.test_net(val_samples, log_file)

            epoch_end = datetime.now()

            print("   -  Epoch %d: Train[acc %5.3f, Loss %5.3f] Validation[acc %5.3f, Loss %5.3f]  -- time: %s"
                  %(epoch+1, epoch_acc[epoch], epoch_loss[epoch], val_acc[epoch], val_loss[epoch], str(epoch_end - epoch_st)))

            with open(log_file, 'a') as log:
                log.write("   -  Epoch %d: Train[acc %5.3f, Loss %5.3f] Validation[acc %5.3f, Loss %5.3f]  -- time: %s/n"
                  %(epoch+1, epoch_acc[epoch], epoch_loss[epoch], val_acc[epoch], val_loss[epoch], str(epoch_end - epoch_st)))

        self.save_model()

        return epoch_acc, epoch_loss

    def test_net(self, all_batches, log_file):
        loss = 0
        acc = 0

        with torch.no_grad():
            for i, batch in enumerate(all_batches, 0):
                if self.cuda:
                    inputs, labels = batch.to('cuda')
                else:
                    inputs, labels = batch

                prediction = self.net(inputs)

                # loss
                batch_loss = self.criterion(prediction, labels)
                loss += batch_loss.item()

                # acc
                max_indx = torch.max(prediction, 1)[1].tolist()
                batch_acc = [1 if max_indx[j] == labels[j] else 0 for j in range(labels.shape[0])].count(1)
                batch_acc /= labels.shape[0]
                acc += batch_acc

        loss /= i
        acc /= i

        print("   -  Accuracy %5.3f, Loss %5.3f" % (acc, loss))
        with open(log_file, 'a') as log:
            log.write("   -  Accuracy %5.3f, Loss %5.3f\n" % (acc, loss))

        return acc, loss

    def save_model(self):
        torch.save(self.net.state_dict(), self.out_dir+"net.h5")

class neural_network(nn.Module):
    def __init__(self, net_param, cuda=False):
        super(neural_network, self).__init__()

        self.base_net = nn.Sequential()
        for i, (inp_s, out_s, k_s) in enumerate(net_param['cnn']):
            layer = nn.Conv2d(inp_s, out_s, kernel_size=k_s)
            self.base_net.add_module("CNN" + str(i + 1), layer)

            if i != 0:
                layer = nn.Dropout2d()
                self.base_net.add_module("drop-out" + str(i + 1), layer)

            layer = nn.MaxPool2d(2)
            self.base_net.add_module("max_pool" + str(i + 1), layer)

            layer = nn.ReLU()
            self.base_net.add_module("Relu" + str(i + 1), layer)

        self.l1 = nn.Linear(net_param['linear'][0][0], net_param['linear'][0][1])
        self.l2 = nn.Linear(net_param['linear'][1][0], net_param['linear'][1][1])

        if cuda:
            self.base_net.to("cuda")
            self.l1.to("cuda")
            self.l2.to("cuda")

        self.init_weights_normal()
        #self.init_weights_uniform

    def init_weights_uniform(self):
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(-initrange, initrange)

            if isinstance(m, nn.Linear):
                nn.init.uniform_(-initrange, initrange)
                nn.init.constant_(m.bias, 0)

    def init_weights_normal(self):
        mean = 0.0
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        input = batch
        hid_out = self.base_net(input).view(input.size(0), -1)
        hid_out = F.relu(self.l1(hid_out))
        hid_out = F.dropout(hid_out, training=self.training)
        hid_out = self.l2(hid_out)
        prediction = F.log_softmax(hid_out)

        return prediction


