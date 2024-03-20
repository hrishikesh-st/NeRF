import os
import cv2
import time
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from NeRFModel import NeRFModel
from logger import Logger
from skimage.metrics import structural_similarity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)


def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix
        images: images
        pose: corresponding camera pose in world frame
    """

    if mode == 'train':
        json_path = os.path.join(data_path, 'transforms_train.json')
    elif mode == 'test':
        json_path = os.path.join(data_path, 'transforms_test.json')
    elif mode == 'val':
        json_path = os.path.join(data_path, 'transforms_val.json')

    print(f"Loading data from {json_path} for {mode}")
    with open(json_path) as file:
        data = json.load(file)

    data_len = len(data['frames'])
    poses = np.zeros((data_len, 4, 4))
    images = []

    for i in range(data_len):
        _pose = np.array(data['frames'][i]['transform_matrix'])
        poses[i] = _pose

        img_file = data['frames'][i]['file_path'] + '.png'
        img_rel_path = os.path.relpath(img_file, '.')
        img_path = os.path.join(data_path, img_rel_path)
        _image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255.
        image = cv2.resize(_image, (400, 400), interpolation = cv2.INTER_AREA)

        if image.shape[2] == 4: # RGBA --> RGB
            image = image[..., :3] * image[..., -1:]  + (1 - image[..., -1:])

        images.append(image[None, ...])

    camera_angle_x = data["camera_angle_x"]
    focal_length = image.shape[0] / (2 * np.tan(0.5 * camera_angle_x))

    images = np.concatenate(images, axis=0)
    print(f"Images shape: {images.shape}")

    return focal_length, images, poses


def PixelToRay(focal_length, poses, images):
    """
    Input:
        camera_info: image width, height, camera matrix
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get tn and tf range, sample rate ...
    Outputs:
        ray origin and direction
    """

    N, H, W, _ = images.shape

    # Rays characterised by origin, direction and target pixel values
    ray_origins = np.zeros((N, H*W, 3))
    ray_directions = np.zeros((N, H*W, 3))
    target_pixel_values = images.reshape((N, H*W, 3))

    # Iterate over each image
    for i in range(N):
        c2w = poses[i]

        u = np.arange(W)
        v = np.arange(H)

        # Generate rays
        u, v = np.meshgrid(u, v)

        # Directions
        dirs = np.stack((u - W / 2, -(v - H / 2), - np.ones_like(u) * focal_length), axis=-1)

        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        ray_directions[i] = dirs.reshape(-1, 3)
        ray_origins[i] += c2w[:3,  3]

    return ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3), target_pixel_values.reshape(-1, 3)


def generateBatch(images, poses, focal_length, batch_size, shuffle=False):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        focal_length: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """

    ray_origins, ray_directions, target_pixel_values = PixelToRay(focal_length, poses, images)
    all_rays = np.concatenate((ray_origins, ray_directions, target_pixel_values), axis=-1)
    return all_rays


def render_scene(neural_field, origins, directions, tn=2, tf=6, samples=192, clear_bg=True):
    """
    Render the scene by querying the neural field across a set of rays.

    Args:
        neural_field: Trained NeRF model for scene rendering.
        origins: The starting points of each ray.
        directions: The direction vectors for each ray.
        tn: Near bounds.
        tf: Far bounds.
        samples: Number of points sampled along each ray.
        clear_bg: If True, render with a white background.

    Returns:
        The rendered RGB colors for each input ray.
    """

    def compute_light_transmission(atten_factors):
        """
        Calculate the cumulative light transmission for each ray segment.
        """
        transmission = torch.cumprod(atten_factors, dim=1)
        return torch.cat((torch.ones(transmission.shape[0], 1, device=transmission.device), transmission[:, :-1]), dim=-1)


    # Define the ray segments for sampling.
    intervals = torch.linspace(tn, tf, samples).to(DEVICE).expand(origins.shape[0], samples)

    # Introduce randomness to the sampling process for anti-aliasing.
    midpoints = (intervals[:, :-1] + intervals[:, 1:]) / 2.0
    lower_bounds = torch.cat((intervals[:, :1], midpoints), -1)
    upper_bounds = torch.cat((midpoints, intervals[:, -1:]), -1)
    random_offsets = torch.rand(intervals.shape, device=DEVICE)
    sampled_points = lower_bounds + (upper_bounds - lower_bounds) * random_offsets

    # Compute the deltas for transmittance calculation.
    intervals_delta = torch.cat((sampled_points[:, 1:] - sampled_points[:, :-1], 
                                 torch.tensor([1e10], device=DEVICE).expand(origins.shape[0], 1)), -1)

    # Determine the 3D coordinates for each sample along the rays.
    points_3D = origins.unsqueeze(1) + sampled_points.unsqueeze(-1) * directions.unsqueeze(1)

    # Query the neural field model for colors and densities at the sampled points.
    sampled_colors, densities = neural_field(points_3D.reshape(-1, 3), 
                                             directions.expand(samples, points_3D.shape[0], 3).transpose(0, 1).reshape(-1, 3))
    sampled_colors = sampled_colors.view(points_3D.shape[0], samples, 3)
    densities = densities.view(points_3D.shape[0], samples)

    # Calculate the alpha values and respective weights for each ray sample.
    alphas = 1 - torch.exp(-densities * intervals_delta)
    weights = compute_light_transmission(1 - alphas).unsqueeze(2) * alphas.unsqueeze(2)

    # Integrate the weighted colors to compute the final ray colors.
    if clear_bg:
        ray_colors = (weights * sampled_colors).sum(1)
        total_weights = weights.sum(dim=[1, 2])
        return ray_colors + (1 - total_weights).unsqueeze(-1)
    else:
        return (weights.unsqueeze(-1) * sampled_colors).sum(1)


def get_loss(groundtruth, prediction):
    """
    Calculate the loss and PSNR between the ground truth and predicted images.

    Args:
        groundtruth: Ground truth pixel values.
        prediction:  Predicted pixel values.

    Returns:
        The mean squared error loss and peak signal-to-noise ratio.
    """

    mse2psnr = lambda x : -10. * torch.log(x).to(DEVICE) / torch.log(torch.Tensor([10.])).to(DEVICE)

    loss = ((prediction - groundtruth)**2).mean()
    psnr = mse2psnr(loss)

    return loss, psnr


def train(data_path, mode='train', num_epochs=1, batch_size=1024, tn=2, tf=6, n_samples=100, lr=5e-4, gamma=0.5,
          pos_freq=10, dir_freq=4, log_every=100, save_every=100, test_every=100, log_dir='trial_logs', verbose=False):
    """
    Training regime for the NeRF model.

    Args:
        data_path: Dataset path.

    Keyword Args:
        mode: Data mode (train/val/test). (default: {'train'})
        num_epochs: Number of training epochs. (default: {1})
        batch_size: Number of rays per batch. (default: {1024})
        tn: Near bound. (default: {2})
        tf: Far bound. (default: {6})
        n_samples: Number of samples per ray. (default: {100})
        lr: Learning rate. (default: {5e-4})
        gamma: Learning rate decay factor. (default: {0.5})
        pos_freq: Positional encoding frequency. (default: {10})
        dir_freq: Directional encoding frequency. (default: {4})
        log_every: Log the results every <> iterations. (default: {100})
        save_every: Save model every <> iterations. (default: {100})
        test_every: Test model every <> iterations. (default: {100})
        log_dir: Directory to save logs. (default: {'trial_logs'})
        verbose: If True, print training progress. (default: {False})
    """

    print("Loading training data...")
    focal_length, images, poses = loadDataset(data_path, mode)

    model = NeRFModel(embed_pos_L=pos_freq, embed_direction_L=dir_freq).to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=gamma)
    _best_loss = 1000

    all_rays = generateBatch(images, poses, focal_length, batch_size, shuffle=True)
    num_batches = int(len(all_rays)/batch_size)

    for epoch in range(num_epochs):

        for i in range(num_batches):
            _idx = np.random.choice(np.arange(len(all_rays)), batch_size)
            batch = all_rays[_idx]

            _start = time.time()

            rays_origin = torch.tensor(batch[:, :3]).to(DEVICE)
            rays_direction = torch.tensor(batch[:, 3:6]).to(DEVICE)
            target_pixel_values = torch.tensor(batch[:, 6:]).to(DEVICE)

            # Forward pass
            predicted_pixel_values = render_scene(model, rays_origin, rays_direction, tn, tf, n_samples)
            loss, psnr = get_loss(target_pixel_values, predicted_pixel_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print(f'Iteration: {i}, Train Loss: {loss.item()}, PSNR: {psnr.item()}, Avg time: {time.time()-_start}')

            if i % log_every == 0:
                logger.log(tag='train', epoch=epoch, iter=i, loss=loss.item(), psnr=psnr.item(), time=time.time()-_start)
                logger.log(tag='msg', epoch=epoch, iter=i, msg='Performing Validation...')
                val_loss = val(model, data_path, mode='val', epoch=epoch, log_every=log_every, verbose=verbose)

                if val_loss < _best_loss:
                    _best_loss = val_loss
                    logger.log(tag='model_loss', loss=_best_loss)
                    torch.save(model.state_dict(), log_path+'/'+log_dir+'/model/best_model.pt')

                logger.log(tag='plot')

            if i % save_every == 0:
                logger.log(tag='model', loss=loss.item())
                torch.save(model.state_dict(), log_path+'/'+log_dir+'/model/model_'+str(epoch)+'_'+str(i)+'.pt')

            if i % test_every == 0:
                test(data_path, mode='test', log_dir=log_dir, img_name='image_'+str(epoch)+'_'+str(i)+'.png')

        scheduler.step()


def val(model, data_path, epoch, mode='val', batch_size=1024, tn=2, tf=6, samples=192, log_every=100, verbose=False):
    """
    Validation regime for the NeRF model.

    Args:
        model: Model to be validated.
        data_path: Dataset path.
        epoch: Current epoch.

    Keyword Args:
        mode: Data mode (train/val/test). (default: {'val'})
        batch_size: Number of rays per batch. (default: {1024})
        tn: Near bound. (default: {2})
        tf: Far bound. (default: {6})
        samples: Number of samples per ray. (default: {192})
        log_every: Log the results every <> iterations. (default: {100})
        verbose: If True, print validation progress. (default: {False})

    Returns:
        Average validation loss.
    """

    print("Loading validation data...")
    focal_length, images, poses = loadDataset(data_path, mode)

    _img_idx = np.arange(len(images))
    _rand_idx = np.random.choice(_img_idx, 4)
    _images = images[_rand_idx]
    _poses = poses[_rand_idx]

    print('Validation Indices:  ', _rand_idx)

    all_rays = generateBatch(_images, _poses, focal_length, batch_size, shuffle=True)
    num_batches = int(len(all_rays)/batch_size)
    avg_loss = 0.0

    for i in range(num_batches):
        _idx = np.random.choice(np.arange(len(all_rays)), batch_size)
        batch = all_rays[_idx]

        _start = time.time()

        rays_origin = torch.tensor(batch[:, :3]).to(DEVICE)
        rays_direction = torch.tensor(batch[:, 3:6]).to(DEVICE)
        target_pixel_values = torch.tensor(batch[:, 6:]).to(DEVICE)

        predicted_pixel_values = render_scene(model, rays_origin, rays_direction, tn, tf, samples)
        loss, psnr = get_loss(target_pixel_values, predicted_pixel_values)
        avg_loss += loss.item()

        if verbose and i % 100 == 0:
            print(f'Iteration: {i}, Val Loss: {loss.item()}, PSNR: {psnr.item()}, Avg time: {time.time()-_start}')

        if i % log_every == 0:
            logger.log(tag='val', epoch=epoch, iter=i, loss=loss.item(), psnr=psnr.item(), time=time.time()-_start)

    avg_loss /= (i+1)

    return avg_loss


def test(data_path, mode='test', image_idx=0, batch_size=1024, tn=2, tf=6, samples=192, H=400, W=400, log_dir='trial_logs', img_name='test.png'):
    """
    Testing regime for the NeRF model.

    Args:
        data_path: Dataset path.

    Keyword Args:
        mode: Data mode (train/val/test). (default: {'test'})
        image_idx: Index of the image to be tested. (default: {0})
        batch_size: Number of rays per batch. (default: {1024})
        tn: Near bound. (default: {2})
        tf: Far bound. (default: {6})
        samples: Number of samples per ray. (default: {192})
        H: Height of the image. (default: {400})
        W: Width of the image. (default: {400})
        log_dir: Directory to save logs. (default: {'trial_logs'})
        img_name: Name of the image to be saved. (default: {'test.png'})
    """
    print("Loading testing data...")
    focal_length, images, poses = loadDataset(data_path, mode)

    all_rays = generateBatch(np.expand_dims(images[image_idx], axis=0), np.expand_dims(poses[image_idx], axis=0), focal_length, batch_size, shuffle=False)

    model = NeRFModel(embed_pos_L=10, embed_direction_L=4).to(DEVICE)
    model.load_state_dict(torch.load(log_path+'/'+log_dir+'/model/best_model.pt'))
    model.eval()
    pixel_values = []
    _psnr = 0.0

    for i in range(0, len(all_rays), batch_size):
        batch = all_rays[i:i+batch_size]

        rays_origin = torch.tensor(batch[:, :3]).to(DEVICE)
        rays_direction = torch.tensor(batch[:, 3:6]).to(DEVICE)
        target_pixel_values = torch.tensor(batch[:, 6:]).to(DEVICE)

        predicted_pixel_values = render_scene(model, rays_origin, rays_direction, tn, tf, samples)
        loss, psnr = get_loss(target_pixel_values, predicted_pixel_values)
        pixel_values.append(predicted_pixel_values.detach().cpu())

        if psnr > 1e10: continue
        _psnr += psnr.item()

    img = torch.cat(pixel_values).numpy().reshape(H, W, 3)*255.0

    ssim = (structural_similarity(img[:, :, 0], images[image_idx][:, :, 0]*255.0, data_range=img[:, :, 0].max()-img[:, :, 0].min()) + 
            structural_similarity(img[:, :, 1], images[image_idx][:, :, 1]*255.0, data_range=img[:, :, 1].max()-img[:, :, 1].min()) +
            structural_similarity(img[:, :, 2], images[image_idx][:, :, 2]*255.0, data_range=img[:, :, 2].max()-img[:, :, 2].min()))/3
    psnr = _psnr * batch_size/len(all_rays)

    print(f'Testing on image index: {image_idx}, PSNR: {psnr}, SSIM: {ssim}')

    cv2.imwrite(log_path+'/'+log_dir+'/media/'+img_name, img)


def test_single_image(tn=2, tf=6, samples=192, batch_size=256, H=400, W=400, log_dir='trial_logs'):
    """
    Testing regime for the NeRF model on a single image.

    Keyword Args:
        tn: Near bound. (default: {2})
        tf: Far bound. (default: {6})
        samples: Number of samples per ray. (default: {192})
        batch_size: Number of rays per batch. (default: {256})
        H: Height of the image. (default: {400})
        W: Width of the image. (default: {400})
        log_dir: Directory to save logs. (default: {'trial_logs'})

    Returns:
        The rendered RGB colors for the input ray.
    """
    # Camera parameters
    focal_length = 555.5555155968841

    # Load the model
    model = NeRFModel(embed_pos_L=10, embed_direction_L=4).to(DEVICE)
    # model.load_state_dict(torch.load('lego_400_19_epochs.pt'))
    model.load_state_dict(torch.load('ship_400_19_epochs.pt'))
    model.eval()
    pixel_values= []

    # Plot rays
    import matplotlib.pyplot as plt
    def plot_rays(o, d, t):
        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        # Label axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        pt1 = o
        pt2 = o + t * d

        for p1, p2 in zip(pt1[::50], pt2[::50]):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

        plt.show()

    trans_t = lambda t : np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ], dtype=np.float32)

    rot_phi = lambda phi : np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1],
    ], dtype=np.float32)

    rot_theta = lambda th : np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1],
    ], dtype=np.float32)


    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w

    frames = []
    count = 0
    for th in tqdm(np.linspace(0., 360., 10, endpoint=False)):
        count += 1
        c2w = pose_spherical(th, -30., 4.)
        rays_origin, rays_direction, _ = PixelToRay(focal_length, np.array([c2w]), np.zeros((1, H, W, 3)))
        all_rays = np.concatenate((rays_origin, rays_direction), axis=-1)

        # Plot rays
        plot_rays(rays_origin, rays_direction, 6)

        pixel_values = []
        for i in tqdm(range(0, len(all_rays), batch_size)):
            batch = all_rays[i:i+batch_size]

            rays_origin = torch.tensor(batch[:, :3]).to(DEVICE)
            rays_direction = torch.tensor(batch[:, 3:6]).to(DEVICE)

            predicted_pixel_values = render_scene(model, rays_origin, rays_direction, tn, tf, samples)
            pixel_values.append(predicted_pixel_values.detach().cpu())

        img = torch.cat(pixel_values).numpy().reshape(H, W, 3)*255.0
        frames.append(img)
        cv2.imwrite(f"image_{count}.png", img)


def main(args):

    global log_path
    log_path = 'Logs'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    global logger
    logger = Logger(os.path.join(log_path, args.log_dir))

    logger.log(tag='args', data_path=args.data_path, num_epochs=args.num_epochs, batch_size=args.batch_size, tn=args.tn, tf=args.tf, 
               samples=args.n_samples, lr=args.lr, gamma=args.gamma, pos_freq=args.n_pos_freq, dir_freq=args.n_dirc_freq)

    if args.mode == 'train':
        print("Start training")
        train(args.data_path, args.mode, args.num_epochs, args.batch_size, args.tn, args.tf, args.n_samples, args.lr, args.gamma,
              args.n_pos_freq, args.n_dirc_freq, args.log_every, args.save_every, args.test_every, args.log_dir, args.verbose)
    elif args.mode == 'test':
        print("Start testing")
        test(args.data_path, args.mode)
    elif args.mode == 'test_single_image':
        test_single_image()


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="Data/lego/", help="dataset path")
    parser.add_argument('--mode', type=str, default='train', help="train/test/test_single_image/val")
    parser.add_argument('--log_dir', type=str, default="trial_logs", help="Logs Directory")
    parser.add_argument('--ckpt_path', type=str, default='trial_logs', help='Checkpoint path to test model')

    parser.add_argument('--num_epochs', type=int, default=20, help="number of epochs for training")
    parser.add_argument('--lr', type=float, default=5e-4, help="training learning rate")
    parser.add_argument('--gamma', type=float, default=0.5, help="decay rate for learning rate scheduler")
    parser.add_argument('--n_pos_freq', type=int, default=10, help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq', type=int, default=4, help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--batch_size', type=int, default=1024, help="number of rays per batch")
    parser.add_argument('--tn', type=int, default=2, help='tn Plane Distance')
    parser.add_argument('--tf', type=int, default=6, help='tf Plane Distance')
    parser.add_argument('--n_samples', type=int, default=192, help="number of samples per ray")
    parser.add_argument('--log_every', type=int, default=1000, help='Log the results every <> iterations')
    parser.add_argument('--save_every', type=int, default=2000, help='Save model every <> iterations')
    parser.add_argument('--test_every', type=int, default=2000, help='Test model every <> iterations')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose Execution')

    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)