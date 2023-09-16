from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
import tifffile
real_image = tifffile.imread("/home/ameyasu/cuda_ws/src/SyMBac/SyMBac/00000.tif")

def main(media_multiplier=75, cell_multiplier=1.7, device_multiplier=29, sigma=8.85,
                                scene_no=-1, match_fourier=False, match_histogram=True, match_noise=False,
                                debug_plot=True, MATLAB_access=True, defocus=3.0, save_dir=".", radius=50, wavelength=0.75, numerical_aperture=1.2, refractive_index=1.3, pix2mic=0.065, apodisation_sigma=20):
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
    my_renderer.select_intensity_MATLAB_access(save_dir = ".", cell_label=None, device_label=None, media_label=None, initialized = True)

    noisy_img, real_resize, mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error, device_var_error = my_renderer.generate_test_comparison(
                                    media_multiplier=media_multiplier, cell_multiplier=cell_multiplier, device_multiplier=device_multiplier, sigma=sigma,
                                    scene_no=scene_no, match_fourier=match_fourier, match_histogram=match_histogram, match_noise=match_noise,
                                    debug_plot=debug_plot, MATLAB_access=MATLAB_access, defocus=defocus, save_dir=save_dir)

    return noisy_img, real_resize, mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error, device_var_error

noisy_img, real_resize, mean_error, media_error, cell_error, device_error, mean_var_error, media_var_error, cell_var_error, device_var_error = main(media_multiplier, cell_multiplier, device_multiplier, sigma,
                                scene_no, match_fourier, match_histogram, match_noise,
                                debug_plot, MATLAB_access, defocus, save_dir, radius, wavelength, numerical_aperture, refractive_index, pix2mic, apodisation_sigma)