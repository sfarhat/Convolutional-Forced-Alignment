import torch
import torch.nn.functional as F

class Sequential_GRAD_CAM(object):

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

    def generate_cam(self, input_shape, target_classes):
        """Generates Class Activation Map using final convolutional layer. If multiple target classes provided,
        it will take the element-wise maximum of all the respective CAM's and return the result. 

        Args:
            target_classes_info (list): List of desired output classes (+ timing info) to compute CAM for, will be combined if length > 1,
                                        each entry has form ([p_class, p_class, p_class], {'start': start, 'end': end}) 
            input_shape (tuple): Shape of input to network which should match desired interpolated CAM dimensions (only height and width) 

        Returns:
            Tensor: Class Activation Map of shape input_shape
        """

        cams = []

        for t, tc in target_classes.items():

            # retain_graph allows for multiple backward() calls
            tc.backward(retain_graph=True)

            # Gradient and activations have shape batch x channels x features x time

            # Global average pool gradients to get weights
            # Needs to maintain dimensions for torch.mul to broadcast correctly
            alpha = torch.mean(self.gradient, dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
            # Alternate way: F.adaptive_avg_pool2d(self.gradient, 1), no need for unsqueezing

            # Weight activations by said weights and ReLU
            # IMPORTANT Algorithm tweak: use time-slices of activation one timestep at a time
            # unsqueeze time dimension since it gets squeezed when slicing for one column blah blah blah
            g_cam = F.relu(torch.sum(torch.mul(self.activation[:,:,:,t].unsqueeze(-1), alpha), dim=1, keepdim=True))

            # Pad in time dimension with zeros on left of start and right of end
            padding = (t, input_shape[1] - (t + 1))
            padded_gcam = F.pad(g_cam, padding, 'constant', 0)
            # Interpolate in features dimension
            cams.append(F.interpolate(padded_gcam, size=input_shape, mode='bilinear'))

        combined_cams = torch.zeros_like(cams[0])

        # This max works since CAM's are ReLU'd and allows for superimposing multiple cams
        for cam in cams:
            # There seems to be high variance among cams that are combined, 
            # so certain classes will make other classes' cams disappear. Normalizing
            # will alleviate this problem
            if torch.all(cam == 0):
                # Sometimes ReLU'd activations are all 0's, will cause NaN problems when normalized
                normalized_cam = cam
            else:
                normalized_cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
            combined_cams = torch.maximum(combined_cams, normalized_cam)

        return combined_cams 

    def get_target_classes(self, log_probs, guessed_labels, desired_phone_indices):
        """Given indices of desired phonemes in guessed_labels, will return time-indexed dictionary of gradient-required objects corresponding to them for use in CAM"""

        def get_target_classes_for_single_phone(log_probs, guessed_labels, desired_phone_idx):
            # Off by one so that first "change" corresponds to first phoneme
            changes = -1
            prev = None
            target_classes_start, target_classes_end = -1, -1
            desired_phoneme = None
            started = False

            for i in range(len(guessed_labels)):
                p = guessed_labels[i]
                if p != prev:
                    if started:
                        target_classes_end = i
                        break
                    changes += 1
                    prev = p

                if changes == desired_phone_idx:
                    if not started:
                        target_classes_start = i
                        started = True
                    desired_phoneme = p

            print('---------------------------')
            print('Range of CAM\'ed label: {} to {}'.format(target_classes_start, target_classes_end))
            print('---------------------------')
            # log_probs is of shape time x classes
            target_classes = {} 

            for t in range(target_classes_start, target_classes_end):
                target_classes[t] = log_probs[t][desired_phoneme] 

            return target_classes

        # {timestep: target_class} 
        target_classes = {} 

        for desired_phone_idx in desired_phone_indices:
            tc = get_target_classes_for_single_phone(log_probs, guessed_labels, desired_phone_idx)
            target_classes.update(tc)

        return target_classes