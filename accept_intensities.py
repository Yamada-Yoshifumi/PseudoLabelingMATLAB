from SyMBac.for_seg_simulation import ForSegSimulation
#from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
from SyMBac.misc import get_sample_images, misc_load_img
import tifffile


def main(real_image, device_label, cell_label, media_label, trench_length=12, trench_width=1.3, cell_length=3, cell_length_var=0.2, cell_width=1, cell_width_var=0.01, radius=50, wavelength=0.75, numerical_aperture=1.2, refractive_index=1.3, pix2mic=0.065, apodisation_sigma=20):

    my_simulation = ForSegSimulation(
        trench_length=trench_length,
        trench_width=trench_width,
        cell_max_length=cell_length, #6, long cells # 1.65 short cells
        cell_width= cell_width, #1 long cells # 0.95 short cells
        opacity_var=0.1,
        sim_length = 50,
        pix_mic_conv = pix2mic,
        max_length_var = cell_length_var,
        width_var = cell_width_var,
        lysis_p= 0.,
        save_dir="./",
        resize_amount = 3,
        show_window=False
    )

    my_simulation.run_simulation()

    my_simulation.draw_simulation_OPL(do_transformation=True, label_masks=False, dynamics_free=True, timeseries_repo = "./")

    my_kernel = PSF_generator(
        radius = radius,
        wavelength = wavelength,
        NA = numerical_aperture,
        n = refractive_index,
        resize_amount = 3,
        pix_mic_conv = pix2mic,
        apo_sigma = apodisation_sigma,
        mode="phase contrast",
        condenser = "Ph3")
    my_kernel.calculate_PSF()
    my_camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)
    #my_camera.render_dark_image(size=(300,300))

    import pickle
    with (open("." + "/simulation.p", "rb")) as openfile:
        try:
            _my_simulation = pickle.load(openfile)
        except EOFError:
            raise EOFError
    my_renderer = Renderer(simulation = _my_simulation, PSF = my_kernel, real_image = real_image, camera = my_camera)
    min_sigma, scene_no = my_renderer.select_intensity_MATLAB_access(save_dir = ".", initialized = False, media_label=media_label, cell_label=cell_label, device_label=device_label)
    #print(min_sigma)
    #print(scene_no)
    return min_sigma, scene_no

real_image = tifffile.imread("/home/ameyasu/cuda_ws/src/SyMBac/SyMBac/00000.tif")
min_sigma, scene_no = main(real_image, device_label, cell_label, media_label, trench_length, trench_width, cell_length, cell_length_var, cell_width, cell_width_var, radius, wavelength, numerical_aperture, refractive_index, pix2mic, apodisation_sigma)