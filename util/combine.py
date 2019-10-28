from pyson.common import *

# ---------------------
output_dir = 'datasets/drug_char_seg/combine/'
os.makedirs(output_dir, exist_ok=True)
def get_name(path):
    name = os.path.basename(path)
    print(name)
    return name.split('.')[0]
    
paths_a  = {get_name(path):path for path in  glob('datasets/drug_char_seg/images/*/*.png')}
paths_b  = glob('./datasets/drug_char_seg/label1/*.png')
paths_b += glob('./datasets/drug_char_seg/label2/*/*.png')
paths_b = {get_name(path):path for path in paths_b}


for name, path_a in tqdm(paths_a.items()):
    if name in paths_b:
        path_b = paths_b[name]
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        img = np.concatenate([img_a, img_b], axis=1)
        
        out_path = os.path.join(output_dir, name+'.png')
        cv2.imwrite(out_path, img)


