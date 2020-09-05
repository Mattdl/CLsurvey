import torch.nn as nn


class AlexNet_LwF(nn.Module):
    def __init__(self, model, last_layer_name=6):
        super(AlexNet_LwF, self).__init__()
        self.model = model
        self.last_layer_name = last_layer_name
        self.finetune_mode = False

    def set_finetune_mode(self, mode):
        self.finetune_mode = mode

    def forward(self, x):
        sub_index = 0
        last_layer = False
        for name, module in self.model._modules.items():
            for namex, modulex in module._modules.items():
                if last_layer:
                    outputs.append(modulex(x))
                else:
                    x = modulex(x)
                if name == 'classifier' and namex == str(self.last_layer_name - 1):
                    last_layer = True
                    outputs = []

            # for reshaping the fully connected layers
            # need to be changed for
            if sub_index == 0:
                x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
            sub_index += 1

        # In this case only return for one head an output
        if self.finetune_mode:
            # print("FINETUNING MODE: outputting single value")
            assert len(outputs) == 1
            outputs = outputs[0]
        return outputs
