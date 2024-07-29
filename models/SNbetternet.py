import torch
import torchvision
import torchvision.utils
import torch.nn as nn
import numpy as np
from PIL import Image
import onnxruntime



class SiameseNetwork(nn.Module):
    def __init__(self, lastLayer=False, pretrained=True):

        super(SiameseNetwork, self).__init__()

        self.lastLayer = lastLayer
        self.net_parameters = []  # list of parameters to be optimized

        self.model_conv = torchvision.models.alexnet(pretrained=pretrained)

        if pretrained:  
            # freeze all parameters in the model
            for param in self.model_conv.parameters():
                param.requires_grad = False

            #  Unfreeze model last layer
            self.out_last = self.model_conv.classifier[6].out_features
            for param in self.model_conv.classifier[6].parameters():
                param.requires_grad = True
                self.net_parameters.append(param)
                #torch.nn.init.xavier_uniform(param)
        else:
            self.out_last = self.model_conv.classifier[6].out_features
            for param in self.model_conv.parameters():
                param.requires_grad = True
                self.net_parameters.append(param)
                #torch.nn.init.xavier_uniform(param)

        if self.lastLayer:
            self.extraL = nn.Linear(self.out_last, 1)
            for param in self.extraL.parameters():
                param.requires_grad = True
                self.net_parameters.append(param)

        self.feature_extractor = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.out_last + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        # loading neural hash model
        self.nh_session = onnxruntime.InferenceSession('/Users/qinxutan/Documents/htxinternship/logorecognition/neuralhash_model.onnx')
        nh_seed = open('/Users/qinxutan/Documents/htxinternship/logorecognition/neuralhash_128x96_seed1.dat', 'rb').read()[128:]
        nh_seed = np.frombuffer(nh_seed, dtype=np.float32)
        nh_seed = nh_seed.reshape([96, 128])
        self.nh_seed = torch.from_numpy(nh_seed)

    def forward_once(self, x):
        output = self.model_conv(x)
        return output
    
    def get_neural_hash(self, im):
        try:
            im = Image.fromarray(im)
            img = im.convert('RGB')
            image = img.resize([360, 360])

            arr = np.array(image).astype(np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            arr = arr.transpose(2, 0, 1)
            arr = arr.reshape([1, 3, 360, 360])

            inputs = {self.nh_session.get_inputs()[0].name: arr}
            outs = self.nh_session.run(None, inputs)

            hash_output = self.nh_seed.dot(outs[0].flatten())
            hash_bits = ''.join('1' if it >= 0 else '0' for it in hash_output)
            hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)

            return hash_hex
        except Exception:
            return "NULL"

    def forward(self, input1, input2, feature_vector):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        diff = torch.abs(output1 - output2)

        feature_embedding = self.feature_extractor(feature_vector)

        combined = torch.cat((diff, feature_embedding), dim=1)
        scores = self.classifier(combined)
        
        return scores
        
    def calculate_hash_similarity(self, hash1, hash2):
        # Example: Calculate similarity using Hamming distance
        if hash1 == "NULL" or hash2 == "NULL":
            return 0.0

        # Calculate Hamming distance
        distance = sum(1 for x, y in zip(hash1, hash2) if x != y)
        similarity = 1.0 - (distance / len(hash1))

        return similarity