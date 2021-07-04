import torch
import torch.nn.functional as F

class GRAD_CAM(object):

    def __init__(self, net):
        self.net = net
        self.activation = None
        self.gradient = None

        def get_activation(model, input, output):
            """Forward hook that returns output at chosen layer of network"""
            self.activation = output.detach()

        def get_gradients(module, grad_in, grad_out):
            """Backwards hook that returns gradient wrt layer of network"""
            self.gradient = grad_out[0].detach()

        self.net.cnn_layers[-1].register_forward_hook(get_activation)
        self.net.cnn_layers[-1].register_backward_hook(get_gradients)

    def generate_cam(self, target_classes, input_shape):
        """Generates Class Activation Map using final convolutional layer. If multiple target classes provided,
        it will take the element-wise maximum of all the respective CAM's and return the result. 

        Args:
            target_classes (list): List of desired output classes to compute CAM for, will be combined if length > 1
            input_shape (tuple): Shape of input to network which should match desired interpolated CAM dimensions (only height and width) 

        Returns:
            Tensor: Class Activation Map of shape input_shape
        """

        cams = []

        for target_class in target_classes:

            # retain_graph allows for multiple backward() calls
            target_class.backward(retain_graph=True)

            # Gradient and activations have shape batch x channels x features x time

            # Global average pool gradients to get weights
            # Needs to maintain dimensions for torch.mul to broadcast correctly
            alpha = torch.mean(self.gradient, dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
            # Alternate way: F.adaptive_avg_pool2d(self.gradient, 1), no need for unsqueezing

            # Weight activations by said weights and ReLU
            g_cam = F.relu(torch.sum(torch.mul(self.activation, alpha), dim=1, keepdim=True))

            # Interpolate
            cams.append(F.interpolate(g_cam, size=input_shape, mode='bilinear'))

        combined_cams = torch.empty_like(cams[0])

        # This max works since CAM's are ReLU'd and allows for superimposing multiple cams
        for cam in cams:
            combined_cams = torch.maximum(combined_cams, cam)

        return combined_cams 

    def get_target_classes(self, log_probs, guessed_labels, desired_phone_idx):
        """Given index of desired phoneme in guessed_labels, will return list of gradient-required objects corresponding to them for use in CAM later"""

        # Off by one so that first "change" corresponds to first phoneme
        changes = -1
        prev = None
        target_class_start, target_class_end = -1, -1
        desired_phoneme = None
        started = False

        for i in range(len(guessed_labels)):
            p = guessed_labels[i]
            if p != prev:
                if started:
                    target_class_end = i
                    break
                changes += 1
                prev = p

            if changes == desired_phone_idx:
                if not started:
                    target_class_start = i
                    started = True
                desired_phoneme = p

        # log_probs is of shape time x classes
        target_class = []
        target_class_timestep_probs = log_probs[target_class_start:target_class_end]
        for probs in target_class_timestep_probs:
            target_class.append(probs[desired_phoneme])

        return target_class