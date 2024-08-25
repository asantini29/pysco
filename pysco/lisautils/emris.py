import numpy as np

from eryn.backends import HDFBackend
from eryn.utils import TransformContainer

from ..eryn import SamplesLoader
from ..utils import find_files


emri_standard_parameter_transforms = {
        0: np.exp,  # M 
        1: np.exp,  # mu
        5: np.arccos, # qS
        7: np.arccos,  # qK
    }

emri_standard_labels = [r"$M \, [M_\odot]$", r"$\mu \, [M_\odot]$", r"$a$", r"$p_0 \, [M]$", r"$d_L \, \rm[Gpc]$", r"$\theta_S$", r"$\phi_S$", r"$\theta_K$", r"$\phi_K$", r"$\Phi_{\phi_0}$"] # I have to generalize this to eccentric orbits

def get_emri_transforms(parameter_transforms='std', fill_dict=None):
    if parameter_transforms == 'std':
        parameter_transforms = emri_standard_parameter_transforms
    else:
        assert isinstance(parameter_transforms, dict)
    
    return TransformContainer(parameter_transforms=parameter_transforms, fill_dict=fill_dict)

class EMRISamplesLoader(SamplesLoader):
    """
    A class for loading EMRI samples.
    Args:
        path (str): The path to the samples.
        parameter_transforms (str, optional): The parameter transforms to apply. Defaults to 'std'.
        fill_dict (dict, optional): A dictionary of values to fill missing parameters. Defaults to None.
    """

    def __init__(self, path, parameter_transforms='std', fill_dict=None, labels=None, extra_labels=[]):

        if labels is None:
            base_labels = emri_standard_labels
        elif isinstance(labels, list):
            base_labels = labels
        elif labels == 'intrinsic':
            base_labels = emri_standard_labels[:4] + emri_standard_labels[-1:] # M, mu, a, p_0, Phi_{\phi_0}
        
        self.labels = base_labels + extra_labels

        super(EMRISamplesLoader, self).__init__(path, transform_fn=get_emri_transforms(parameter_transforms, fill_dict))

    def get_injection(self):
        """
        Get the injections parameters.
        Returns:
            dict: The injections parameters.
        """
        injection_file = find_files(self.path, 'npy')[0]
        injection = self.transform_fn.both_transforms(np.load(injection_file))[:, None]

        assert injection.shape[0] == len(self.labels), f"Expected {len(self.labels)} parameters, got {injection.shape[0]}."

        dict(zip(self.labels, injection))


        