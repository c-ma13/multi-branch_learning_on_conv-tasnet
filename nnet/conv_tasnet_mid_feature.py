#  import sys
#  sys.path.append("..")

import torch
import torch.nn as nn

model_info = {
    "exp_name": "whamr",
    "model_name": "conv-tasnet",
    "loss_type": "sisdr",
    "model_param": {
        "num_spk": 2,
        "enc_dim": 256,#512
        "feature_dim": 256,#128 #line151,*4
        "win": 20,
        "layer": 8,
        "stack": 3,
        "unshared_stack_in": 1,
        "unshared_stack_out": 1,
        "kernel": 3,
    }
}


# deleted causal keyword
class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True):
        super(DepthConv1d, self).__init__()
        
        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
                                 groups=hidden_channel,
                                 padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        
        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


# deleted causal keyword
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, unshared_stack_in,unshared_stack_out, kernel=3, skip=True, 
                 dilated=True):
        super(TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # normalization
        self.LN1 = nn.GroupNorm(1, input_dim, eps=1e-8)
        self.LN2 = nn.GroupNorm(1, input_dim, eps=1e-8)

        self.BN1 = nn.Conv1d(input_dim, BN_dim, 1)
        self.BN2 = nn.Conv1d(input_dim, BN_dim, 1)
        
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN_in1 = nn.ModuleList([])
        self.TCN_in2 = nn.ModuleList([])
        for s in range(unshared_stack_in):
            for i in range(layer):
                if self.dilated:
                    self.TCN_in1.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip)) 
                    self.TCN_in2.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip)) 
                else:
                    self.TCN_in1.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip))
                    self.TCN_in2.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip))   

        self.TCN = nn.ModuleList([])
        for s in range(stack-unshared_stack_in-unshared_stack_out):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip)) 
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip))   
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
        self.TCN_out1 = nn.ModuleList([])
        self.TCN_out2 = nn.ModuleList([])
        for s in range(unshared_stack_out):
            for i in range(layer):
                if self.dilated:
                    self.TCN_out1.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip)) 
                    self.TCN_out2.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip)) 
                else:
                    self.TCN_out1.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip))
                    self.TCN_out2.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip))   
                    
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
        # output layer
        
        self.output1 = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )
        self.output2 = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )
        
        self.skip = skip
        
    def forward(self, input_r, input_a):
        
        # input shape: (B, N, L)
        
        # normalization
        output_r = self.BN1(self.LN1(input_r))
        output_a = self.BN2(self.LN2(input_a))
        
        # pass to TCN
        if self.skip:
            skip_connection_r = 0.
            skip_connection_a = 0.
            for i in range(len(self.TCN_in1)):
                residual_r, skip_r = self.TCN_in1[i](output_r)
                output_r = output_r + residual_r
                skip_connection_r = skip_connection_r + skip_r
                residual_a, skip_a = self.TCN_in2[i](output_a)
                output_a = output_a + residual_a
                skip_connection_a = skip_connection_a + skip_a
            feature_r = output_r
            feature_a = output_a
            for i in range(len(self.TCN)):
                residual_r, skip_r = self.TCN[i](output_r)
                output_r = output_r + residual_r
                skip_connection_r = skip_connection_r + skip_r
                residual_a, skip_a = self.TCN[i](output_a)
                output_a = output_a + residual_a
                skip_connection_a = skip_connection_a + skip_a

            skip_connectionr1 = skip_connection_r
            skip_connectionr2 = skip_connection_r
            skip_connectiona1 = skip_connection_a
            skip_connectiona2 = skip_connection_a
            outputr1 = output_r
            outputr2 = output_r
            outputa1 = output_a
            outputa2 = output_a

            for i in range(len(self.TCN_out1)):
                residual, skip = self.TCN_out1[i](outputr1)
                outputr1 = outputr1 + residual
                skip_connectionr1 = skip_connectionr1 + skip
                residual, skip = self.TCN_out2[i](outputr2)
                outputr2 = outputr2 + residual
                skip_connectionr2 = skip_connectionr2 + skip
                residual, skip = self.TCN_out1[i](outputa1)
                outputa1 = outputa1 + residual
                skip_connectiona1 = skip_connectiona1 + skip
                residual, skip = self.TCN_out2[i](outputa2)
                outputa2 = outputa2 + residual
                skip_connectiona2 = skip_connectiona2 + skip

        else:
            for i in range(len(self.TCN)): # unchanged
                residual = self.TCN[i](output)
                output = output + residual
            output1 = output
            output2 = output
            for i in range(len(self.TCN1)):
                residual = self.TCN1[i](output1)
                output1 = output1 + residual
                residual = self.TCN2[i](output1)
                output2 = output2 + residual
            
        # output layer
        if self.skip:
            outputr1 = self.output1(skip_connectionr1)
            outputr2 = self.output2(skip_connectionr2)
            outputa1 = self.output1(skip_connectiona1)
            outputa2 = self.output2(skip_connectiona2)
        else:
            output1 = self.output1(output1)
            output2 = self.output2(output2)
        
        return outputr1, outputr2, outputa1, outputa2, feature_r, feature_a


# deleted causal keyword
# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, model_info):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = model_info["model_param"]["num_spk"]

        self.enc_dim = model_info["model_param"]["enc_dim"]
        self.feature_dim = model_info["model_param"]["feature_dim"]
        
        self.win = model_info["model_param"]["win"]
        self.stride = self.win // 2
        
        self.layer = model_info["model_param"]["layer"]
        self.stack = model_info["model_param"]["stack"]
        self.unshared_stack_in = model_info["model_param"]["unshared_stack_in"]
        self.unshared_stack_out = model_info["model_param"]["unshared_stack_out"]
        self.kernel = model_info["model_param"]["kernel"]
        
        # input encoder
        self.encoder_1 = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        self.encoder_2 = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        # TCN separator
        self.TCN = TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*2,
                              self.layer, self.stack, self.unshared_stack_in, self.unshared_stack_out, self.kernel)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder1 = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        self.decoder2 = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = torch.autograd.Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input_r, input_a):
        
        # padding
        output_r, rest = self.pad_signal(input_r)
        output_a, rest = self.pad_signal(input_a)
        batch_size = output_r.size(0)
        
        # waveform encoder
        enc_output_r = self.encoder_1(output_r)  # B, N, L
        enc_output_a = self.encoder_2(output_a)  # B, N, L

        # generate masks
        m1, m2, m3, m4, feature_r, feature_a = self.TCN(enc_output_r, enc_output_a)
        masks1 = torch.sigmoid(m1).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masks2 = torch.sigmoid(m2).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_outputr1 = enc_output_r.unsqueeze(1) * masks1  # B, C, N, L
        masked_outputr2 = enc_output_r.unsqueeze(1) * masks2  # B, C, N, L
        masks1 = torch.sigmoid(m3).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masks2 = torch.sigmoid(m4).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_outputa1 = enc_output_a.unsqueeze(1) * masks1  # B, C, N, L
        masked_outputa2 = enc_output_a.unsqueeze(1) * masks2  # B, C, N, L
        
        # waveform decoder
        output1 = self.decoder1(masked_outputr1.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output1 = output1[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output1 = output1.view(batch_size, self.num_spk, -1)  # B, C, T
        output2 = self.decoder2(masked_outputr2.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output2 = output2[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output2 = output2.view(batch_size, self.num_spk, -1)  # B, C, T
        output3 = self.decoder1(masked_outputa1.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output3 = output3[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output3 = output3.view(batch_size, self.num_spk, -1)  # B, C, T
        output4 = self.decoder2(masked_outputa2.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output4 = output4[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output4 = output4.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output1,output2,output3,output4,feature_r,feature_a


if __name__ == "__main__":
    x = torch.rand(3, 32000)
    nnet = TasNet(model_info)
    x,y,z,w = nnet(x, x)
    # s1 = x[:,0]
    # print(s1.shape)
    # print(y.shape)
    print(x.shape,y.shape,z.shape,w.shape)