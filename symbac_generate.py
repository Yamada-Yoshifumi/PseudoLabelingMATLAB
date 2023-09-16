from SyMBac.for_seg_simulation import ForSegSimulation
#from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
from SyMBac.misc import get_sample_images, misc_load_img
import tifffile
real_image = tifffile.imread("/home/ameyasu/cuda_ws/src/SyMBac/SyMBac/00000.tif")

def main(save_dir, n_samples, cell_length=3, cell_length_var=0.2, cell_width=1, cell_width_var=0.01, trench_length=12, trench_width=1.3, radius=50, wavelength=0.75, numerical_aperture=1.2, refractive_index=1.3, pix2mic=0.065, apodisation_sigma=20):

    import pickle

    parameters = {"trench_length":trench_length, 
                  "trench_width":trench_width, 
                  "radius":radius,
                  "wavelength":wavelength,
                  "numerical_aperture": numerical_aperture,
                  "refractive_index": refractive_index,
                  "pix2mic": pix2mic,
                  "apodisation_sigma": apodisation_sigma}

    with open("./simulation_parameters.p", "wb") as f:
        pickle.dump(parameters, f)

    my_simulation = ForSegSimulation(
        trench_length=trench_length,
        trench_width=trench_width,
        cell_max_length=cell_length, #6, long cells # 1.65 short cells
        cell_width= cell_width, #1 long cells # 0.95 short cells
        opacity_var=0.1,
        sim_length = n_samples,
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

    with (open("." + "/simulation.p", "rb")) as openfile:
        try:
            _my_simulation = pickle.load(openfile)
        except EOFError:
            raise EOFError
    my_renderer = Renderer(simulation = _my_simulation, PSF = my_kernel, real_image = real_image, camera = my_camera)
    my_renderer.select_intensity_MATLAB_access(save_dir = ".", cell_label=None, device_label=None, media_label=None, initialized = True)

    my_renderer.update_synth_image_params(save_dir=".")

    my_renderer.generate_training_data(sample_amount=0.1, randomise_hist_match=True, randomise_noise_match=True, burn_in=0, n_samples = int(n_samples), save_dir=save_dir, params_dir = ".", in_series=False)

#if (trench_length is None or trench_width is None or radius is None or wavelength is None or numerical_aperture is None or refractive_index is None or pix2mic is None or apodisation_sigma is None):
try:
    main(save_dir, n_samples, cell_length, cell_length_var, cell_width, cell_width_var, trench_length, trench_width, radius, wavelength, numerical_aperture, refractive_index, pix2mic, apodisation_sigma)
except NameError:
    print("Reusing locally saved parameters")
    with (open("." + "/simulation_parameters.p", "rb")) as openfile:
        try:
            import pickle
            simulation_parameters = pickle.load(openfile)
            main(save_dir, n_samples, cell_length, cell_length_var, cell_width, cell_width_var, simulation_parameters["trench_length"], simulation_parameters["trench_width"], simulation_parameters["radius"], simulation_parameters["wavelength"], simulation_parameters["numerical_aperture"], simulation_parameters["refractive_index"], simulation_parameters["pix2mic"], simulation_parameters["apodisation_sigma"])

        except EOFError:
            raise EOFError