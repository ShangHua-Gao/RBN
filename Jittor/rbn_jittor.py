import jittor.nn as nn
import jittor as jt
from jittor import init, Module

class RepresentativeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, is_train=True, sync=True):
        super(RepresentativeBatchNorm2d, self).__init__(
                num_features, eps, momentum, affine, is_train, sync)
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = init.constant((1, num_features, 1, 1), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((1, num_features, 1, 1), "float32", 0.0) if affine else 0.0
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

        ### weights for centering calibration ###        $
        self.center_weight = init.constant((1, num_features, 1, 1), "float32", 0.0)
        ### weights for scaling calibration ###            $
        self.scale_weight = init.constant((1, num_features, 1, 1), "float32", 1.0)
        self.scale_bias = init.constant((1, num_features, 1, 1), "float32", 0.0)
        ### calculate statistics ###$
        self.stas = nn.AdaptiveAvgPool2d((1,1))

    def execute(self, x):
        dims = [0]+list(range(2,x.ndim))
        ####### centering calibration begin ####### $
        x += self.center_weight*self.stas(x)
        ####### centering calibration end ####### $
        if self.is_train:
            xmean = jt.mean(x, dims=dims)
            x2mean = jt.mean(x*x, dims=dims)
            if self.sync and jt.in_mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = (x2mean-xmean*xmean).maximum(0.0)
            w = 1.0 / jt.sqrt(xvar+self.eps)
            b =  - xmean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)

            self.running_mean.update(self.running_mean +
                (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
            self.running_var.update(self.running_var +
                (xvar.reshape((-1,))-self.running_var)*self.momentum)
            
        else:
            w = 1.0 / jt.sqrt(self.running_var+self.eps)
            b = - self.running_mean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
        
        ####### scaling calibration begin ####### $
        scale_factor = jt.sigmoid(self.scale_weight*self.stas(norm_x)+self.scale_bias)
        ####### scaling calibration end ####### $
        return self.weight*scale_factor*norm_x  + self.bias

