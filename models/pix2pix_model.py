import os
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

import util.util as util
from util.image_pool import ImagePool

from . import networks
from .adamw import AdamW
from .base_model import BaseModel
from .decoder import BeamCTCDecoder, GreedyDecoder


def checkInterVarGrad(grad):
    import numpy as np
    if np.isnan(grad.data.cpu().numpy()).any():
        __import__("ipdb").set_trace()


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.epoch = 1
        self.gan_loss = opt.gan_loss
        self.isTrain = opt.isTrain

        # define tensors self.Tensor has been reloaded
        self.input_A = self.Tensor(opt.batchSize).cuda(device=self.gpu_ids[0])
        self.input_B = self.Tensor(opt.batchSize).cuda(device=self.gpu_ids[0])

        # load/define networks
        self.netG = networks.define_G(gpu_ids=opt.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
            #                              opt.which_model_netD,
            #                              opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.epoch = int(opt.which_epoch)
            self.load_network(self.netG, 'G', opt.which_epoch)
            # if self.isTrain:
            #    self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterion = CTCLoss(size_average=True)

            # initialize optimizers

            self.TrainableParam = list()
            param = self.netG.named_parameters()
            IgnoredParam = [id(P) for name, P in param if 'stft' in name]

            if self.opt.optimizer == 'Adam':
                self.optimizer_G = torch.optim.Adam(
                       self.netG.parameters(),
                        lr=opt.lr,
                        betas=(0.9, 0.999), weight_decay=self.opt.weightDecay)


            if self.opt.optimizer == 'AdamW':
                self.optimizer_G = AdamW(
                       self.netG.parameters(),
                        lr=opt.lr,
                        betas=(0.9, 0.999), weight_decay=self.opt.weightDecay)



            if self.opt.optimizer == 'sgd':
                self.optimizer_G = torch.optim.SGD(
                    filter(lambda P: id(P) not in IgnoredParam,
                       self.netG.parameters()),
                        lr=opt.lr)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, 'min', factor=0.25, patience=35, verbose=True)

            self.decoder = BeamCTCDecoder('_0123456789', num_processes=196, beam_width=100, cutoff_top_n=4000)
            ##TODO
            self.greedyDecoder = GreedyDecoder('_0123456789')


            print('---------- Networks initialized ---------------')
            networks.print_network(self.netG)
            # networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):

        ## TODO: auto map member
        AtoB = self.opt.which_direction == 'AtoB'
        self.input_A = input['inputs'].cuda()
        self.inputPercentages = input['inputPercentages']
        self.labels = input['labels']
        self.labelSize = input['labelSize']
        self.inputInfo = input['fileInfo']

    def forward(self):
        self.realA = Variable(self.input_A, requires_grad=False)
        output = self.netG.forward(self.realA)

        self.fakeB = output
        #self.realB = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.netG.eval()
        self.realA = Variable(self.input_A, volatile=True)
        out = self.netG.forward(self.realA)
        seqLen = out.size(1)
        sizes = self.inputPercentages.mul_(int(seqLen)).int()
        split_targets = []
        offset = 0

        for size in self.labelSize:
            try:
                split_targets.append(self.labels[offset:offset + size])
            except:
                __import__("ipdb").set_trace()
            offset += size

        decoded_output, _ = self.decoder.decode(out.data, sizes)
        target_strings = self.greedyDecoder.convert_to_strings(split_targets)
        wer, cer = 0, 0

        flag = open('/home/caspardu/oracle', 'r').read().strip()
        if flag == 'd': 
            __import__('ipdb').set_trace()

        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer += self.decoder.wer(transcript, reference) / float(len(reference.split(' ')))
            cer += self.decoder.cer(transcript, reference) / float(len(reference))

        batchSize = len(target_strings)

        self.netG.train() 
        return {'wer': wer / batchSize, 'cer': cer / batchSize}

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fakeB
        fake_AB = self.fake_AB_pool.query(
            torch.cat((self.real_A, self.fakeB), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.realB), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        ## self.fakeB: Batch x Time x nClasses
        prob = self.fakeB.transpose(0, 1)
        seqLen = prob.size(0)

        labels = Variable(self.labels, requires_grad=False)
        labelSize = Variable(self.labelSize, requires_grad=False)
        sizes = Variable(self.inputPercentages.mul_(int(seqLen)).int(), requires_grad=False)
        self.loss_G = self.criterion(prob, labels, sizes, labelSize)


        if not np.isinf(self.loss_G.cpu().data.numpy()):
            self.loss_G.backward()
        else:
            #__import__('ipdb').set_trace()
            pass
        
            

    def optimize_parameters(self):
        self.netG.train()
        self.forward()

        if self.gan_loss:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def annealLR(self, loss):
        self.scheduler.step(loss)

    def get_current_errors(self):
        if self.gan_loss:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0])])
        else:
            # print("#############clean sample mean#########")
            # sample_data = self.input_B.cpu().numpy()

            # print("max value", np.max(sample_data))
            # print("mean value", np.mean(np.abs(sample_data)))
            return OrderedDict([('G_LOSS', self.loss_G.data.sum())])
            # return self.loss_G.data[0]

    def get_current_visuals(self):
        real_A = self.real_A.data.cpu().numpy()
        fakeB = self.fakeB.data.cpu().numpy()
        realB = self.realB.data.cpu().numpy()
        clean = self.clean.cpu().numpy()
        noise = self.noise.cpu().numpy()
        return OrderedDict([
            ('est_ratio', fakeB),
            ('clean', clean),
            ('ratio', realB),
            ('noise', noise),
        ])

    def save(self):
        label = self.epoch
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.gan_loss:
            self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        # lrd = self.opt.lr / self.opt.niter_decay
        # lr = self.old_lr - lrd
        lr = self.old_lr * 0.6
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    def visSaliency(self):
        def hookFunction(module, grad_in, grad_out):
         self.gradients = grad_in

        def get_positive_negative_saliency(gradient):
         pos_saliency = (np.maximum(0, gradient) / gradient.max())
         neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
         return pos_saliency, neg_saliency

        self.netG.zero_grad()
        input_ = Variable(self.input_A, requires_grad=True)
        output_ = self.netG.forward(input_)
        self.netG._modules['module']._modules['conv']._modules['0'].register_backward_hook(hookFunction)
        self.forward()
        one_hot = torch.FloatTensor(output_.size()).zero_()
        classId = 5
        one_hot[::,::,classId] = 1
        output_.backward(gradient=one_hot)
        gradArr = np.sum(input_.grad.data.cpu().numpy()[0], axis=0)
        posS, negS = get_positive_negative_saliency(gradArr)
        from skimage import io, transform

        import matplotlib
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        import matplotlib.colors as mcolors
        def transparent_cmap(cmap, N=255):
         "Copy colormap and set alpha values"
         mycmap = cmap
         mycmap._init()
         mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
         return mycmap
        mycmap = transparent_cmap(plt.cm.spring)
        def plotHeat(oriImg, posS, index):
         y, x = np.mgrid[0:posS.shape[0], 0:posS.shape[1]]
         fig, ax = plt.subplots(1, 1)
         ax.imshow(oriImg)
         cb = ax.contourf(x, y, posS / np.max(posS), 15, cmap=mycmap)
         plt.colorbar(cb)
         plt.savefig('heat/miserab1e_{}.png'.format(index))

        os.system('rm -rf heat heat.7z')
        os.system('mkdir heat')
#posS = np.clip(posS, 0., 0.1) * 4.9999
        posS = np.clip(posS, 0., 0.999) 
        pos = np.log1p(posS)
        for i in range(posS.shape[0]):
         oriImg = transform.resize(io.imread(os.path.join(self.inputInfo[0], os.listdir(self.inputInfo[0])[i])), posS[0].shape)
         plotHeat(oriImg, posS[i], i)

        os.system("7z a heat.7z heat")
