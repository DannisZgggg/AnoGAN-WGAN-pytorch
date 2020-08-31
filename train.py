"""AnoGAN
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##

import shutil
import cv2
from tqdm import tqdm
import warnings
from options import Options

import torch.optim as optim
from evaluate import *

## TODO: choose dataset and model here >>>
from model import NetG, NetD #_64
from data.data_mnist import provider

##
class AnoGAN:
    """AnoGAN Class
    """
    def __init__(self, opt):
        # super(AnoGAN, self).__init__(opt, dataloader)

        # Initalize variables.
        self.opt = opt

        self.niter = self.opt.niter
        self.start_iter = 0
        self.netd_niter = 5
        self.test_iter = 100
        self.lr = self.opt.lr
        self.batchsize = {'train': self.opt.batchsize, 'test': 1}

        self.pretrained = False

        self.phase = 'train'
        self.outf = self.opt.experiment_group
        self.algorithm = 'wgan'

        # LOAD DATA SET
        self.dataloader = {'train':provider('train',opt.category,batch_size=self.batchsize['train'],num_workers=4),
                           'test' :provider('test',opt.category,batch_size=self.batchsize['test'],num_workers=4)}

        self.trn_dir = os.path.join(self.outf, self.opt.experiment_name, 'train')
        self.tst_dir = os.path.join(self.outf, self.opt.experiment_name, 'test')

        self.test_img_dir = os.path.join(self.outf, self.opt.experiment_name, 'test', 'images')
        if not os.path.isdir(self.test_img_dir):
            os.makedirs(self.test_img_dir)

        self.best_test_dir = os.path.join(self.outf, self.opt.experiment_name, 'test', 'best_images')
        if not os.path.isdir(self.best_test_dir):
            os.makedirs(self.best_test_dir)

        self.weight_dir = os.path.join(self.trn_dir, 'weights')
        if not os.path.exists(self.weight_dir): os.makedirs(self.weight_dir)

        # -- Misc attributes
        self.epoch = 0

        self.l_con = l1_loss
        self.l_enc = l2_loss

        ##
        # Create and initialize networks.
        self.netg = NetG().cuda()
        self.netd = NetD().cuda()

        # Setup optimizer
        self.optimizer_d = optim.RMSprop(self.netd.parameters(), lr=self.lr)
        self.optimizer_g = optim.Adam(self.netg.parameters(),lr=self.lr)

        ##
        self.weight_path = os.path.join(self.outf, self.opt.experiment_name, 'train', 'weights')
        if os.path.exists(self.weight_path) and len(os.listdir(self.weight_path)) == 2:
            print("Loading pre-trained networks...\n")
            self.netg.load_state_dict(torch.load(os.path.join(self.weight_path, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.weight_path, 'netD.pth'))['state_dict'])

            self.optimizer_g.load_state_dict(torch.load(os.path.join(self.weight_path, 'netG.pth'))['optimizer'])
            self.optimizer_d.load_state_dict(torch.load(os.path.join(self.weight_path, 'netD.pth'))['optimizer'])

            self.start_iter = torch.load(os.path.join(self.weight_path, 'netG.pth'))['epoch']


    ##
    def start(self):
        """ Train the model
        """

        ##
        # TRAIN
        # self.total_steps = 0
        best_criterion = -1 #float('inf')
        best_auc = -1

        # Train for niter epochs.
        # print(">> Training model %s." % self.name)
        for self.epoch in range(self.start_iter, self.niter):
            # Train for one epoch
            mean_wass = self.train()

            (auc, res, best_rec, best_threshold), res_total = self.test()
            message = ''
            # message += 'criterion: (%.3f+%.3f)/2=%.3f ' % (best_rec[0], best_rec[1], res)
            # message += 'best threshold: %.3f ' % best_threshold
            message += 'Wasserstein Distance:%.3d ' % mean_wass
            message += 'AUC: %.3f ' % auc

            print(message)

            torch.save(
                {'epoch': self.epoch + 1,
                 'state_dict': self.netg.state_dict(),
                 'optimizer': self.optimizer_g.state_dict()}, '%s/netG.pth' % (self.weight_dir))

            torch.save(
                {'epoch': self.epoch + 1,
                 'state_dict': self.netd.state_dict(),
                 'optimizer': self.optimizer_d.state_dict()}, '%s/netD.pth' % (self.weight_dir))

            if auc > best_auc:
                best_auc = auc
                new_message = "******** New optimal found, saving state ********"
                message = message + '\n' + new_message
                print(new_message)

                for img in os.listdir(self.best_test_dir):
                    os.remove(os.path.join(self.best_test_dir, img))

                for img in os.listdir(self.test_img_dir):
                    shutil.copyfile(
                        os.path.join(self.test_img_dir,img),
                        os.path.join(self.best_test_dir,img)
                )

                shutil.copyfile(
                    '%s/netG.pth' % (self.weight_dir),
                    '%s/netg_best.pth' % (self.weight_dir)
                )


            log_name = os.path.join(self.outf, self.opt.experiment_name, 'loss_log.txt')
            message = 'Epoch%3d:' % self.epoch + ' ' + message
            with open(log_name, "a") as log_file:
                if self.epoch == 0:
                    log_file.write('\n\n')
                log_file.write('%s\n' % message)


        print(">> Training %s Done..." % self.opt.experiment_name)

    ##
    def train(self):
        """ Train the model for one epoch.
        """
        print("\n>>> Epoch %d/%d, Running " % (self.epoch + 1, self.niter) + self.opt.experiment_name)

        self.netg.train()
        self.netd.train()
        # for p in self.netg.parameters(): p.requires_grad = True

        mean_wass = 0

        tk0 = tqdm(self.dataloader['train'], total=len(self.dataloader['train']))
        for i, itr in enumerate(tk0):
            input, _ = itr
            input = input.cuda()
            wasserstein_d = None
            # if self.algorithm == 'wgan':
            # train NetD
            for _ in range(self.netd_niter):
                # for p in self.netd.parameters(): p.requires_grad = True
                self.optimizer_d.zero_grad()

                # forward_g
                latent_i = torch.rand(self.batchsize['train'],64,1,1).cuda()
                fake = self.netg(latent_i)
                # forward_d
                _, pred_real = self.netd(input)
                _, pred_fake = self.netd(fake)  # .detach() TODO

                # Backward-pass
                wasserstein_d = (pred_fake.mean() - pred_real.mean()) * 1
                wasserstein_d.backward()
                self.optimizer_d.step()

                for p in self.netd.parameters(): p.data.clamp_(-0.01,0.01) #<<<<<<<

            # train netg
            # for p in self.netd.parameters(): p.requires_grad = False
            self.optimizer_g.zero_grad()
            noise = torch.rand(self.batchsize['train'], 64, 1, 1).cuda()
            fake = self.netg(noise)
            _, pred_fake = self.netd(fake)
            err_g_d = - pred_fake.mean()  # negative

            err_g_d.backward()
            self.optimizer_g.step()

            errors = {
                'loss_netD': wasserstein_d.item(),
                'loss_netG': round(err_g_d.item(), 3),
            }

            mean_wass += wasserstein_d.item()
            tk0.set_postfix(errors)

            if i % 50 == 0:
                img_dir = os.path.join(self.outf, self.opt.experiment_name, 'train', 'images')
                if not os.path.isdir(img_dir):
                    os.makedirs(img_dir)
                self.save_image_cv2(input.data, '%s/reals.png' % img_dir)
                self.save_image_cv2(fake.data, '%s/fakes%03d.png' % (img_dir,i))

        mean_wass /= len(self.dataloader['train'])
        return mean_wass

    ##
    def test(self):
        """ Test AnoGAN model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        self.netg.eval()
        self.netd.eval()
        # for p in self.netg.parameters(): p.requires_grad = False
        # for p in self.netd.parameters(): p.requires_grad = False

        for img in os.listdir(self.test_img_dir):
            os.remove(os.path.join(self.test_img_dir, img))


        self.phase = 'test'
        meter = Meter_AnoGAN()
        tk1 = tqdm(self.dataloader['test'], total=len(self.dataloader['test']))
        for i,itr in enumerate(tk1):
            input, target = itr
            input = input.cuda()

            latent_i = torch.rand(self.batchsize['test'], 64, 1, 1).cuda()
            latent_i.requires_grad = True

            optimizer_latent = optim.Adam([latent_i], lr=self.lr)
            test_loss = None
            for _ in range(self.test_iter):
                optimizer_latent.zero_grad()
                fake = self.netg(latent_i)
                residual_loss = self.l_con(input, fake)
                latent_o, _ = self.netd(fake)
                discrimination_loss = self.l_enc(latent_i, latent_o)
                alpha = 0.1
                test_loss = (1 - alpha) * residual_loss + alpha * discrimination_loss
                test_loss.backward()
                optimizer_latent.step()

            abnormal_score = test_loss
            meter.update(abnormal_score,target) #<<<TODO

            # Save test images.
            combine = torch.cat([input.cpu(),fake.cpu()],dim=0)
            self.save_image_cv2(combine, '%s/%05d.jpg' % (self.test_img_dir, i + 1))


        criterion, res_total = meter.get_metrics()

        # rename images
        for i, res in enumerate(res_total):
            os.rename('%s/%05d.jpg' % (self.test_img_dir, i + 1),
                      '%s/%05d_%s.jpg' % (self.test_img_dir, i + 1, res))


        return criterion, res_total

    @staticmethod
    def save_image_cv2(tensor, filename):
        # return
        from torchvision.utils import make_grid
        # tensor = (tensor + 1) / 2
        grid = make_grid(tensor, 8, 2, 0, False, None, False)
        ndarray = grid.mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv2.imwrite(filename, ndarray)



if __name__ == '__main__':
    import random
    seed_value = 47
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    warnings.filterwarnings("ignore")


    opt = Options().parse()
    model = AnoGAN(opt)
    model.start()
