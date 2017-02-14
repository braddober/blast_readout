import numpy as nm
import scipy
import scipy.fftpack
import matplotlib.pyplot as plt

def amplitude_and_power_spectrum( timestream, tau, return_amplitudes=False ):
    # in the scanned pdf notes
    # timestream is v
    # yfft is v-tilde

    # delete the last data point if the timestream has an odd number of data points
    if nm.mod(len(timestream),2):
        timestream = timestream[0:-1]
        # gently warn that this has happened
        print 'Note: final data point deleted for FFT'

    # number of data points in the voltage timestream
    N = len(timestream)

    # take fft, shift it, and normalize with the factor of N
    yfft = scipy.fftpack.fft(timestream)
    yfft = scipy.fftpack.fftshift(yfft)
    yfft = (1.0/N)*yfft

    # make frequency array for fft
    delta = 1.0/(N*tau)
    nyquist = 1.0/(2.0*tau)
    freq = nm.arange(-nyquist,nyquist,delta) # note that the last element will be (nyquist-delta) because of python's conventions

    # make positive frequency array
    pos_freq = freq[(N/2):]
    # because of roundoff error, the zeroth frequency bin appears to be 1e-12 or something
    # fix it to zero by definition
    pos_freq[0] = 0.0

    # as an intermediate step, normalize the FFT such that sum(psd) = rms(timestream)
    temp = yfft[(N/2):] # positive freq half of fft
    psd = temp*nm.conj(temp)
    psd = 2.0*psd
    psd[0] = psd[0]/2.0 # special case for DC

    if return_amplitudes:
        # Amplitude and Phase spectrum
        # calculate amplitudes from the psd variable
        amplitudes = nm.sqrt(2.0*psd) # root-two comes from the conversion from rms to peak-to-peak amplitude
        amplitudes[0] = nm.sqrt(psd[0]) # special case for DC
        # calculate phase angles from yfft
        phases = nm.arctan2(nm.imag(yfft[(N/2):]),nm.real(yfft[(N/2):]))
        return pos_freq,amplitudes,phases   
    else:
        # V / sqrt(Hz) spectrum
        v_sqrt_hz = nm.sqrt(psd / delta)
        # since it's zero, throw away imaginary part
        v_sqrt_hz = nm.real(v_sqrt_hz)
        return pos_freq,v_sqrt_hz

def multi_plot():
	phases_1000 = nm.load('./phases_1000.npy')
	phases_500 = nm.load('./phases_500.npy')
	phases_250 = nm.load('./phases_250.npy')
	phases_100 = nm.load('./phases_100.npy')
	v_sqrtHz_1000 = nm.zeros((1000, len(phases_1000[0])/2))
	v_sqrtHz_500 = nm.zeros((500, len(phases_500[0])/2))
	v_sqrtHz_250 = nm.zeros((250, len(phases_250[0])/2))
	v_sqrtHz_100 = nm.zeros((100, len(phases_100[0])/2))

	for chan in range(1000): 
		freqs, v_sqrtHz_1000[chan] = amplitude_and_power_spectrum(phases_1000[chan], 1/244.14)
		#print nm.sum(phases_1000[chan]**2/len(phases_1000[chan])), nm.sum( v_sqrtHz_1000[chan]**2 * (freqs[1] - freqs[0]))  
	for chan in range(500): 
		freqs, v_sqrtHz_500[chan] = amplitude_and_power_spectrum(phases_500[chan], 1/244.14)
	for chan in range(250): 
		freqs, v_sqrtHz_250[chan] = amplitude_and_power_spectrum(phases_250[chan], 1/244.14)
	for chan in range(100): 
		freqs, v_sqrtHz_100[chan] = amplitude_and_power_spectrum(phases_100[chan], 1/244.14)

	PSD_1000 = 10*nm.log10(nm.mean(v_sqrtHz_1000**2, axis = 0))
	PSD_500 = 10*nm.log10(nm.mean(v_sqrtHz_500**2, axis = 0))
	PSD_250 = 10*nm.log10(nm.mean(v_sqrtHz_250**2, axis = 0))
	PSD_100 = 10*nm.log10(nm.mean(v_sqrtHz_100**2, axis = 0))

	figure = plt.figure(num=1, figsize = (20,12), dpi=200, facecolor='w', edgecolor='w')
	ax1 = figure.add_subplot(1,1,1)
	ax1.set_xscale('log')
	ax1.set_ylim((-130,-70))
	ax1.set_xlim((0.01, nm.max(freqs)))
	ax1.set_ylabel('dBc/Hz', size = 28)
	ax1.set_xlabel('log Hz', size = 28)
	ax1.tick_params(axis='x', labelsize=25, width = 2, which = 'both') 
	ax1.tick_params(axis='y', labelsize=25, width = 2, which = 'both') 
	plt.grid()
	ax1.plot(freqs, PSD_100, color = 'b', alpha = 0.7, linewidth = 1, label = '100')
	ax1.plot(freqs, PSD_250, color = 'm', alpha = 0.7, linewidth = 1, label = '250')
	ax1.plot(freqs, PSD_500, color = 'r', alpha = 0.6, linewidth = 1, label = '500')
	ax1.plot(freqs, PSD_1000, color = 'g', alpha = 0.5, linewidth = 1, label = '1000')
	plt.legend(loc = 'upper right', fontsize = 28)
	plt.savefig('/home/muchacho/sam/jaif7.pdf', bbox = 'tight')
	plt.show()
	return 

def nist_data():
	phase_off = nm.load('/home/muchacho/upenn_summer_2016/phase_off.npy')
	phase_on = nm.load('/home/muchacho/upenn_summer_2016/phase_on.npy')
	phase_rotated = nm.load('/home/muchacho/upenn_summer_2016/phase_rot.npy')
	accum_freq = 244.14
        Npackets = nm.shape(phase_on)[0]
	v_sqrtHz_off = nm.zeros((574, Npackets/2))
	v_sqrtHz_on = nm.zeros((574, Npackets/2))
	v_sqrtHz_rot = nm.zeros((574, Npackets/2))
	plot_range = (Npackets / 2) + 1
	figure = plt.figure(num= None, figsize = (20,12), dpi=200, facecolor='w', edgecolor = 'w')
	ax = figure.add_subplot(1,1,1)
	ax.set_xscale('log')
	ax.set_ylim((-100,-20))
        plt.xlim((0.01, accum_freq/2.))
        ax.set_ylabel('dBc/Hz', size = 28)
        ax.set_xlabel('log Hz', size = 28)
        ax.tick_params(axis='x', labelsize=25, which = 'both', width = 2)
        ax.tick_params(axis='y', labelsize=25, which = 'both', width = 2)
        plt.grid()
	# do one to get the size of the arrays
	freqs, v_tmp = amplitude_and_power_spectrum(phase_off[:,0], 1/244.14)
	v_sqrthz_rot_ave = nm.zeros_like(v_tmp)
	v_sqrthz_off_ave = nm.zeros_like(v_tmp)
	counter = 0.0
	for chan in range(574): 
		freqs, v_sqrtHz_off[chan] = amplitude_and_power_spectrum(phase_off[:,chan], 1/244.14)
		freqs, v_sqrtHz_on[chan] = amplitude_and_power_spectrum(phase_on[:,chan], 1/244.14)
		freqs, v_sqrtHz_rot[chan] = amplitude_and_power_spectrum(phase_rotated[:,chan], 1/244.14)
		PSD_on = 10*nm.log10(v_sqrtHz_on[chan]**2)
		if (PSD_on[1] < -27) and (PSD_on[100] < -40) and (PSD_on[-1] > -100):
			v_sqrthz_rot_ave += v_sqrtHz_rot[chan] 
			v_sqrthz_off_ave += v_sqrtHz_off[chan]
			counter += 1.0
	v_sqrthz_rot_ave /= counter
	v_sqrthz_off_ave /= counter
	PSD_off = 10*nm.log10(v_sqrthz_off_ave**2)
	PSD_rot = 10*nm.log10(v_sqrthz_rot_ave**2)
	ax.plot(freqs, PSD_rot, color = '#FF4500', label = 'On', linewidth = 1)
	ax.plot(freqs, PSD_off, color = 'k', alpha = 0.8, label = 'Off', linewidth = 1)
	plt.legend(loc = 'lower_left', fontsize = 28)	
	plt.tight_layout()
	plt.savefig('/home/muchacho/sam/new_jaif8.pdf', bbox = 'tight')
	plt.show()
	return
