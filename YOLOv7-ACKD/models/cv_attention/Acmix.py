import typing as t
import torch
import torch.nn as nn


# 生成（1,2，h，w）形状的张量，参与后面位置编码操作
def position(H, W, is_cuda=True,dtype=float):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc.type(dtype=dtype)


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]  # 前两维全取，在h和w的维度上以stride的步长取值


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)  # 将tensor中的值都填充为0.5


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes=10, out_planes=10, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes  # 输入通道
        self.out_planes = out_planes  # 输出通道
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation  # 扩张系数
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        #  torch.nn.ReflectionPad2d；对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        # nn.Unfold（）：将Tensor切割成kernel_size大小的块，输出是（bs,Cxkernel_size[0]xkernel_size[1],L)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=True)

        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)  # 把rate1设置为0.5
        init_rate_half(self.rate2)  # 把rate2设置为0.5
        print(self.rate2.dtype)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)  # 9*3*3
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.  # 将长方体对角线上的设置为1
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)  # 加入位置编码
        init_rate_0(self.dep_conv.bias)  # 把bias设置为0

    def forward(self, x):
        # 通过卷积获取qkv
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)

        scaling = torch.tensor(self.head_dim).pow(-0.5)

        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride  # 等效后的尺寸

        # positional encoding  -1到1上分别有h和w个数的组成的（1，head_dim，h，w）
        pe = self.conv_p(position(h, w, is_cuda=x.is_cuda, dtype=x.dtype))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:  # 在 h，w方向按步长取值
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        # 计算qk张量的点积，求其相似度
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)

        # q*k*v
        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        # conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], dim=1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    kernel_size = input.shape[2]
    acmix = ACmix(in_planes=512, out_planes=512)
    output = acmix(input)
    # print(output.shape)
