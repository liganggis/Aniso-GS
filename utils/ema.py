import torch

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.isStart = False

    def initParam(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone()
        self.isStart = True
    
    def register0(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = torch.zeros_like(param.data)
        self.isStart = True

    def register1(self, batch):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] += param.data.clone() / (255.0*batch)
        
    def update(self, decay):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name][:] = param.clone()  + decay * (self.shadow[name] - param.clone())

    def updateSWA(self, batch):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name][:] = self.shadow[name] + (param.clone() - self.shadow[name]) / (batch + 1)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.clone()
                param[:] = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param[:] = self.backup[name]
        self.backup = {}

    def store(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param[:] = self.shadow[name]

# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#         self.isStart = False

#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()
#         self.isStart = True
    
#     def register0(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = 0

#     def register2(self, batch):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = self.shadow[name] + param.data.clone() / (292.0*batch)
        
#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()

#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data.clone()
#                 param.data = self.shadow[name].clone()

#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name].clone()
#         self.backup = {}