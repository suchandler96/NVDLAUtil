import torch
import torch.nn as nn

from pytorch2caffe import pytorch2caffe


# matrix (dim1 x dim2) x matrix (dim2 x dim3)
class FCWithConv(nn.Module):
    def __init__(self, dim1, dim2, dim3, use_bias):
        super(FCWithConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=dim2, out_channels=dim3, kernel_size=(1, 1), bias=use_bias)

    def forward(self, x):
        return self.conv(x)


# Q * K^T * V
# matrix (dim1 x dim2) x matrix (dim2 x dim3) x matrix (dim3 x dim4)
class DoubleMatmulWithConv(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super(DoubleMatmulWithConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim2, out_channels=dim3, kernel_size=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim3, out_channels=dim4, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


# temporarily useless
class QKVNets(nn.Module):
    def __init__(self, num_tokens, head_dimension):
        super(QKVNets, self).__init__()
        self.q_net = FCWithConv(num_tokens, head_dimension, head_dimension, use_bias=True)
        self.k_net = FCWithConv(num_tokens, head_dimension, head_dimension, use_bias=True)
        self.v_net = FCWithConv(num_tokens, head_dimension, head_dimension, use_bias=True)

    def forward(self, x):
        return self.q_net(x) + self.k_net(x) + self.v_net(x)


class PointWiseNet(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super(PointWiseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim2, out_channels=dim3, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=dim3, out_channels=dim4, kernel_size=(1, 1), bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


def convert_fc_with_conv(seq_len=64, head_dim=128, name="fc_layer_conv"):
    fc_layer = FCWithConv(seq_len, head_dim, head_dim, use_bias=True)
    fc_layer.eval()
    dummy_input = torch.ones([1, head_dim, 1, seq_len])

    pytorch2caffe.trans_net(fc_layer, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


def convert_qkv_nets(seq_len=64, head_dim=128, name="qkv_nets"):
    qkv_nets = QKVNets(num_tokens=seq_len, head_dimension=head_dim)
    qkv_nets.eval()
    dummy_input = torch.ones([1, head_dim, 1, seq_len])

    pytorch2caffe.trans_net(qkv_nets, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


# input matrix dim: seq_len x head_dim
def convert_double_matmul_with_conv(seq_len=64, head_dim=128, name="double_matmul_conv"):
    layer = DoubleMatmulWithConv(seq_len, head_dim, seq_len, head_dim)
    dummy_input = torch.ones([1, head_dim, 1, seq_len])

    pytorch2caffe.trans_net(layer, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


def convert_point_wise_feed_forward_with_conv(seq_len=64, head_dim=128, name="pwn_conv"):
    layer = PointWiseNet(seq_len, head_dim, 4 * head_dim, head_dim)
    dummy_input = torch.ones([1, head_dim, 1, seq_len])

    pytorch2caffe.trans_net(layer, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


if __name__ == "__main__":
    # convert_fc_with_conv(seq_len=128, head_dim=64, name="QNet_B1_L128_HD64")
    # convert_fc_with_conv(seq_len=128, head_dim=64, name="KNet_B1_L128_HD64")
    # convert_fc_with_conv(seq_len=128, head_dim=64, name="VNet_B1_L128_HD64")

    convert_double_matmul_with_conv(seq_len=128, head_dim=64, name="QKTV_B1_L128_HD64")

    # convert_point_wise_feed_forward_with_conv(seq_len=128, head_dim=64, name="PWN_B1_L128_HD64")

    # there is still skip connection, will do it after concatenating all the .txn files
