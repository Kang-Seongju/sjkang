import torch.nn.functional as F

from utils.parse_config import *
from utils.utils import *

# replace False to True for tensorrt trasformation
ONNX_EXPORT = False

def create_modules(module_defs, arc, num_cls):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    width = int(hyperparams['width'])
    height  = int(hyperparams['height'])
    if int(hyperparams['phase']) == 0: 
        phase = "train"
    else:
        phase = "inference"

    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=pad,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'gelu':
                modules.add_module('activation', nn.GELU())

        elif mdef['type'] == 'gelan':
            filters = int(mdef['filters'])
            modules.add_module('gelan', G_ELAN(output_filters[-1], phase))

        elif mdef['type'] == 'attention':
            size = int(mdef['size']) #kernel_size
            filters = int(mdef['filters']) # out_channel
            modules.add_module('attention', patch_wise_attention_layer(in_channel = output_filters[-1], kernel_size= size))
        
        elif mdef['type'] == 'maxpool':
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'before':

            filters = (num_cls + 5) * 3

            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=pad,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=False))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'gelu':
                modules.add_module('activation', nn.GELU())

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(anchors=mdef['anchors'][mask],  # anchor list
                                nc = num_cls,  # number of classes
                                img_size=(height, width),  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1 or 2
                                arc=arc)  # yolo architecture

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs, height, width


class patch_wise_attention_layer(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super(patch_wise_attention_layer, self).__init__()
        # in_channel 은 kernel_size의 배수여야 한다
        self.in_channel = in_channel #
        self.kernel_size = kernel_size
        self.stride = self.kernel_size
        self.padding = 0

        self.patch_block = nn.Conv2d(self.in_channel, self.in_channel, kernel_size = self.kernel_size, stride = self.stride)
        self.unfold = nn.Unfold(3,1,0,1)
        self.fp = nn.ReflectionPad2d(1)

    def forward(self, x):
        # bs , c, h, w
        in_dim = x.shape
        bs, ch, h, w = in_dim
        patch_height = h // self.kernel_size
        patch_width = w // self.kernel_size
        x1 = self.patch_block(x) # [bs, c, kernel_size, kernel_size]
        x = x1.view([bs*ch, -1, patch_height, patch_width]) # 채널 별 패치 간 연산을 위해
        x = self.fp(x)
        x = self.unfold(x)
        x = x.view([bs, ch, 3*3, -1]).permute((0,3,1,2)).contiguous().view([-1, ch, 3*3])
        xT = x.transpose(1,2)[:,0,:].view(-1,1,ch) # 연산량 kernel_size 배 감소
        
        xxT = torch.bmm(xT,x) # 확인

        xxT = xxT.view([bs* patch_height *patch_width, -1])

        xxT = torch.sum(xxT, dim = 1).view([bs, 1, patch_height, patch_width]) # sum or mean 
        xxT = torch.sigmoid(xxT)

        y = xxT.repeat(1, ch, 1, 1)
                
        out = x1*y
        # out dimension = [bs, c, h/k, w/h] downsampling 
        return out


class GaussianDiffusionTrainer(nn.Module):
    # beta_1 ~ beta_T 구간을 T개의 step으로 동일한 간격으로 나누고
    def __init__(self, scale_factor):
        super().__init__()
        self.sf = scale_factor
        
    def forward(self, x_0):
        noise = torch.randn_like(x_0, requires_grad=False)
        x_t = (1. - self.sf)* x_0 + self.sf * noise
        return x_t


class G_ELAN(nn.Module):
    def __init__(self, in_channel):
        super(G_ELAN, self).__init__()
        self.in_channel = in_channel
        self.half = in_channel // 2
        self.GN1 = GaussianDiffusionTrainer(0.07)
        self.GN2 = GaussianDiffusionTrainer(0.05)
        self.GN3 = GaussianDiffusionTrainer(0.10)
        self.GN4 = GaussianDiffusionTrainer(0.08)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = self.half, out_channels = self.half, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.half, momentum = 0.1),
            nn.GELU()
        )
        self.conv1x1 = nn.Conv2d(self.in_channel , self.in_channel, kernel_size = 1, stride = 1 , padding = 0)
       
    def forward(self, x):
        
        # partial
        l = x[:,:self.half,:,:] 
        r = x[:,self.half:,:,:]

        b = self.conv2d(r)
        b = self.GN1(b)
        b = torch.add(b, r)

        b1 = self.conv2d(b)
        b1 = self.GN2(b1)
        b1 = torch.add(b, b1)

        b2 = self.conv2d(b1)
        b2 = self.GN3(b2)
        b2 = torch.add(b1, b2)

        b3 = self.conv2d(b2)
        b3 = self.GN4(b3)
        b3 = torch.add(b2, b3)

        l = self.GN4(l)
        r = self.GN4(r)

        # c = torch.cat([l, r, b1, b3 ], dim = 1)
        # out = self.conv1x1(c) 
        c = torch.add(l,b3)
        d = torch.add(b1,r)
        e = torch.cat([c,d],dim = 1)
        out = self.conv1x1(e) 
        return out

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.oi = [0, 1, 2, 3] + list(range(5, self.no))  # output indices
        self.arc = arc
        s = 256
        ch = 3
        forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
        self.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
        
    def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
        nx, ny = ng  # x and y grid size
        self.img_size = max(img_size)
        self.stride = self.img_size / max(ng)

        # build xy offsets
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(device) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
        self.ng = torch.Tensor(ng).to(device)
        self.nx = nx
        self.ny = ny

    def forward(self, p, img_size):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids(self, img_size, (nx, ny), p.device, p.dtype)
                '''
                img size -> 256, 416
                yolo layer 1  -> (13,8) 
                yolo layer 2 -> (26,16)
                yolo layer 3 ->(52, 32)
                '''
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        '''
        p shape!!!!!!!!
        torch.Size([1, 27 (na x no), 8, 13])
        torch.Size([1, 27, 16, 26])
        torch.Size([1, 27, 32, 52])
        '''
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction


        '''
        torch.Size([1, 3, 8, 13, 9])
        torch.Size([1, 3, 16, 26, 9])
        torch.Size([1, 3, 32, 52, 9])
        '''
        if self.training:
            return p

        elif ONNX_EXPORT:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            m = self.na * self.nx * self.ny
            '''
            self.ng 
            layer 1 => [13, 8]
            layer 2 => [26, 16]
            layer 3 => [52, 32]
            '''
            ngu = self.ng.repeat((1, m, 1))
            '''
            self.grid_xy shape :
            torch.Size([1, 1, 8, 13, 2])
            torch.Size([1, 1, 16, 26, 2])
            torch.Size([1, 1, 32, 52, 2])
            '''
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view(1, m, 2)
            # anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(1, m, 2) / ngu
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(1, m, 2)
            p = p.view(m, self.no) # p ->
            xy = (torch.sigmoid(p[..., 0:2]) + grid_xy[0])* self.stride  # x, y
            wh = (torch.exp(p[..., 2:4]) * anchor_wh[0])* self.stride  # width, height

            p_conf = torch.sigmoid(p[:, 4:5])  # Conf
            p_cls = F.softmax(p[:, 5:self.no], 1) * p_conf  # SSD-like conf
            # a = torch.cat((xy,wh),1)
            # print(a[0])
            '''
            [9, 312]
            [9, 1248]
            [9, 4992]
            '''
            # return torch.cat((xy / ngu[0], wh, p_conf, p_cls), 1).t()

            return torch.cat((xy, wh, p_conf, p_cls), 1).t()

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            if 'default' in self.arc:  # seperate obj and cls
                torch.sigmoid_(io[..., 4:])
            elif 'BCE' in self.arc:  # unified BCE (80 classes)
                torch.sigmoid_(io[..., 5:])
                io[..., 4] = 1
            elif 'CE' in self.arc:  # unified CE (1 background + 80 classes)
                io[..., 4:] = F.softmax(io[..., 4:], dim=4)
                io[..., 4] = 1

            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # compute conf
            io[..., 5:] *= io[..., 4:5]  # conf = obj_conf * cls_conf
            # output :
            # torch.Size([1, 312, 8])   8x13
            # torch.Size([1, 1248, 8])   16x26
            # torch.Size([1, 4992, 8]    32x52
            return io[..., self.oi].view(bs, -1, self.no - 1), p


class CCLAB(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, arc='default', num_cls = 80):
        super(CCLAB, self).__init__()
        self.num_cls = num_cls
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs, self.height, self.width = create_modules(self.module_defs, arc, self.num_cls)
        self.yolo_layers = get_yolo_layers(self)
        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x):
        img_size = x.shape[-2:]

        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool', 'attention', 'gelan', 'before']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layers], print(x.shape)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])

        if self.training:
            return output
        elif ONNX_EXPORT:
            output = torch.cat(output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            nc = self.module_list[self.yolo_layers[0]].nc  # number of classes
            '''
            ONNX scores : output[5:5 + nc].t() -> [6552, 4] # class is  4
            boxes predict : output[0:4].t() -> [6552, 4]
            '''
            return output[5:5 + nc].t(), output[0:4].t()  # ONNX scores, boxes
        else:

            io, p = list(zip(*output))  # inference output, training output

            '''
            io -> [ [1,312,8], [1.1248,8], [1, 4992, 8] ]
            out -> torch.cat(io,1) => size ([1, 6552, 8])
            '''
            return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional' or mdef['type'] ==  'before':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)