import os.path
from data.base_dataset import *
from data.image_folder import make_dataset
from PIL import Image
from pyson.common import *
from pyson.vision import norm_range
from data.aligned_dataset import AlignedDataset

class DrugDataset(AlignedDataset):
    def __init__(self, *args, **kwargs):
        super(DrugDataset, self).__init__(*args, **kwargs)
        self.transform = get_transform_drug(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        A = np.array(A)
        B = AB.crop((w2, 0, w, h))
        B = np.array(B)

        out = self.transform(image=A, mask=B)
        A = torch.from_numpy(out['image']).permute([2,0,1])
        B = torch.from_numpy(out['mask']).permute([2,0,1])
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
