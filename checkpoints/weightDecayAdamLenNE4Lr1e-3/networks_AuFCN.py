class AuFCN(nn.Module):
    def __init__(self, numClasses=11):
        super(AuFCN, self).__init__()
        modList = list()
        modList = [
                nn.Conv3d(3, 64, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                #nn.Dropout3d(),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2)),
                nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.Dropout3d(),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2)),
                nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Dropout3d(),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2))
                ]
        #self.conv = nn.ModuleList(modList)
        self.conv = nn.Sequential(*modList)

        ## TODO ADD OPTIONS

        rnnList = [
                BatchRNN(input_size=4608, hidden_size=256, bidirectional=True, batch_norm=True),
                BatchRNN(input_size=512, hidden_size=256, bidirectional=True, batch_norm=True),
                ]
        self.rnn = nn.Sequential(*rnnList)
        fcBlock = nn.Sequential(
                nn.BatchNorm1d(512),
                #nn.Dropout(),
                nn.Linear(512, numClasses, bias=False)
                )
        self.dense = nn.Sequential(
                SequenceWise(fcBlock)
                )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, sample):
        ## sample shape Batch x Channel x Time x H x w
        ## output of conv shape: Batch x Channel x time x H x w
        output = self.conv(sample)
        ## flatten: batch x time x 4608
        output = output.view(output.size(0), output.size(2), -1)

        ## transpose time x batch x 4608
        output = output.transpose(1, 0)
        try:
            assert output.size(-1) == 4608
        except:
            __import__('ipdb')
        output = self.rnn(output)
        ## output of rnn time x batch x 512
        output = self.dense(output)
        ## output of dense time x batch x nclasses
        output = output.transpose(1, 0)
        ## output batch x time x nclasses
        output = self.inference_softmax(output)
        return output
