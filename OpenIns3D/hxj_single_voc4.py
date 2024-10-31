import os
import sys
import torch
import numpy as np
import open3d as o3d
from openins3d.lookup import Lookup
from openins3d.snap import Snap
import cv2
import pyviz3d.visualizer as viz
from openins3d.mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party/YOLO-World'))
os.environ['ODISE_MODEL_ZOO'] = os.path.expanduser('~/.torch/iopath_cache')

def read_data(path_3D_scans,k):
    name_of_scene = path_3D_scans.split("/")[-1].split(".")[0]
    print("scene name: ", name_of_scene)    
    if path_3D_scans.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(path_3D_scans) 
        mesh = load_mesh(path_3D_scans)
        if(k!=0):
            R = pcd.get_rotation_matrix_from_xyz((k*np.pi / 2, np.pi / 2, 0))  
            pcd.rotate(R, center=(0, 0, 0)) 
            R1 = mesh.get_rotation_matrix_from_xyz((k*np.pi / 2, np.pi / 2, 0))  
            mesh.rotate(R1, center=(0, 0, 0)) 

        pcd_rgb = np.hstack([np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors) * 255.])

        xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)* 255.
        xyz_rgb = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).float()
    elif path_3D_scans.endswith(".npy"):
        xyz_rgb = np.load(path_3D_scans)[:, :6]
    else:
        raise ValueError("Unsupported file format")
    
    return pcd_rgb, xyz_rgb



def single_vocabulary_detection( path_3D_scans, k,vocabulary, path_images = None,detector="odise"):
    pcd_rgb, xyz_rgb=read_data(path_3D_scans,k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("third_party/scannet200_val.ckpt").to(device).eval()
    data, features, _, inverse_map = prepare_data(pcd_rgb, device)
    with torch.no_grad():
        mask= map_output_to_pointcloud(model(data, raw_coordinates=features), inverse_map, 0.5)

    snap_class = Snap([800, 800], [2, 0.5, 1.0], "output/hxj_snap_demo")
    lookup_class = Lookup([800, 800], 0.5, "output/hxj_snap_demo", text_input=vocabulary, results_folder="output/hxj_results_demo")

    name_of_scene1=f"{name_of_scene}_{k}"
    snap_class.scene_image_rendering(xyz_rgb, name_of_scene1,k, mode=["global", "wide", "corner"])

    if detector =="odise":
        lookup_class.call_ODISE()
    elif detector == "yoloworld":
        lookup_class.call_YOLOWORLD()

    if path_images:
        mask_classfication, score = lookup_class.lookup_pipelie(xyz_rgb, mask, name_of_scene1, threshold = 0.6, use_2d=True, single_detection=True)
    else:
        mask_classfication, score = lookup_class.lookup_pipelie(xyz_rgb, mask, name_of_scene1, threshold = 0.6, use_2d=False, single_detection=True)

    mask_final = mask[:, [i for i in range(len(mask_classfication)) if mask_classfication[i] != -1]]
    # # save the results as image
    # snap_class.scene_image_rendering(torch.tensor(xyz_rgb).float(), f"{name_of_scene1}_vis", mode=["global", "wide", "corner"], mask=[mask_final, None])
    # print("Detection compelted. There are {} objects detected.".format(mask_final.shape[1]))
    # return xyz, rgb, mask_final, mask, v

def plot_mask(original_mask, final_mask, scene_coord, scene_color, name_of_scene, v):


    for idx_mask in range(original_mask.shape[1]):
        mask_individual = original_mask[:, idx_mask].bool()
        mask_point = scene_coord[mask_individual]
        mask_color = scene_color.copy()
        mask_final_color = scene_color.copy()
        for i in range(original_mask.shape[1]):
            mask_i = original_mask[:, i]
            mask_i = mask_i.bool()
            random_colr = np.random.rand(3)* 255.
            mask_color[mask_i, :] = random_colr

        for i in range(final_mask.shape[1]):
            mask_i = final_mask[:, i]
            mask_i = mask_i.bool()
            mask_final_color[mask_i, :] = [255, 0, 0]
    v.add_points(f"{name_of_scene}_rgb", scene_coord, scene_color, point_size=20, visible=False)
    v.add_points(f"{name_of_scene}_allmask", scene_coord, mask_color, point_size=20, visible=False)
    v.add_points(f"{name_of_scene}_detected", scene_coord, mask_final_color, point_size=20, visible=True)

if __name__ == "__main__":
    v = viz.Visualizer()

    name_of_scene = "demo_2"
    path_3D_scans = f"/home/hanglok/Desktop/HXJ/code/OpenIns3D/data/hxj/scenes/{name_of_scene}.ply"
    path_masks = None
    path_images = None
    vocabulary = [
        'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 
        'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 
        'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 
        'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 
        'toilet paper', 'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 
        'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 
        'mirror', 'copier', 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 
        'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 
        'recycling bin', 'container', 'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 
        'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 
        'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 
        'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 
        'kitchen counter', 'doorframe', 'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 
        'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 
        'mouse', 'toilet seat cover dispenser', 'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 
        'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 
        'stuffed animal', 'headphones', 'dish rack', 'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 
        'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 
        'alarm clock', 'music stand', 'projector screen', 'divider', 'laundry detergent', 'bathroom counter', 'object', 
        'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 
        'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod', 'coffee kettle', 'structure', 'shower head', 
        'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 
        'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress'
    ]
    detector="odise"
    # xyz, rgb, mask_final, mask, v = single_vocabulary_detection(path_3D_scans,0, vocabulary,  path_images,detector)
    # xyz1, rgb1, mask_final1, mask1, v1 = single_vocabulary_detection(path_3D_scans,1, vocabulary, path_images,detector)
    # xyz2, rgb2, mask_final2, mask2, v2 = single_vocabulary_detection(path_3D_scans,2, vocabulary,  path_images,detector)
    # xyz3, rgb3, mask_final3, mask3, v3 = single_vocabulary_detection(path_3D_scans,3, vocabulary,  path_images,detector)
    # xyz4, rgb4, mask_final4, mask4, v4 = single_vocabulary_detection(path_3D_scans,4, vocabulary, path_images,detector)

    single_vocabulary_detection(path_3D_scans,0, vocabulary,  path_images,detector)
    # single_vocabulary_detection(path_3D_scans,1, vocabulary, path_images,detector)
    # single_vocabulary_detection(path_3D_scans,2, vocabulary,  path_images,detector)
    # single_vocabulary_detection(path_3D_scans,3, vocabulary,  path_images,detector)
    # single_vocabulary_detection(path_3D_scans,4, vocabulary, path_images,detector)

    # plot_mask(mask, mask_final, xyz, rgb, name_of_scene, v)
    # v.save(f'output/hxj_demo/viz')

    