

### This script contains a bunch of functions to simulate the distributions due to weak bursts
from pylab import *
import numpy as np
import scipy.stats
from scipy.stats.mstats import mquantiles as quantiles

import powerspectrum
import word
import parameters
import burstmodel

rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)

def flat_poisson(nsims=10000):

    ## segment length
    tseg = 0.5

    ## time resolution
    dt = 0.5/2048.0

    times = np.arange(0.0, tseg, dt)

    total_counts = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    pbins = [1,5,10,20,50]

    maxpowers = np.zeros((len(total_counts), len(pbins), nsims))
    theoretical_sample = np.zeros((len(total_counts), len(pbins), nsims))

    for i,t in enumerate(total_counts):
        cr = np.ones(len(times))*t*dt/tseg
        maxpowers[i,:,:], theoretical_sample[i,:,:], df = simulate_powers(times, cr, nsims=nsims, pbins=pbins,
                                                                      namestr="flatlc_p%i"%t)

    return

def simulate_powers(times, cr, nsims=10000, pbins=[1,5,10,20,50], fmin=1000.0, namestr="test", plotmean=False,
                    plotlc=False, plothist=False):

    print("mean count rate: " + str(np.mean(cr)))

    maxpowers = np.zeros((len(pbins), nsims))
    theoretical_sample = np.zeros((len(pbins), nsims))


    for n in xrange(nsims):
        #print("simulation %i" %n)
        cpoisson = np.array([np.random.poisson(c) for c in cr])
        if plotlc:
            if n <=5:
                fig = figure(figsize=(24,9))
                ax = fig.add_subplot(121)
                plot(times, cpoisson, lw=2, color="black", linestyle="steps-mid")
                xlabel("Time [s]", fontsize=20)
                ylabel("Counts per bin", fontsize=20)

        ps = powerspectrum.PowerSpectrum(times, counts=cpoisson, norm="leahy")
        df = ps.df
        if plotlc:
            if n<=5:
                ax2 = fig.add_subplot(122)
                loglog(ps.freq[1:], ps.ps[1:],lw =2, color="black", linestyle="steps-mid")
                xlabel("Frequency [Hz]", fontsize=20)
                ylabel("Leahy Power", fontsize=20)
        if plotmean:
            if n == 0:
                psmean = ps.ps
            else:
                psmean += ps.ps

        for j,b in enumerate(pbins):
            if b == 1:
                f_ind = np.array(ps.freq).searchsorted(fmin)
                maxpowers[j,n] = np.max(ps.ps[f_ind:])
                theoretical_sample[j,n] = np.max(np.random.chisquare(2*b, size=len(ps.ps[f_ind:])))/float(b)

            else:
                bps = ps.rebinps(ps.df*b)
                if b == 20:
                    if plotlc:
                        if n<=5:
                            loglog(bps.freq[1:], bps.ps[1:], lw=2, color="red", linestyle="steps-mid")

                f_ind = np.array(bps.freq).searchsorted(fmin)
                maxpowers[j,n] = np.max(bps.ps[f_ind:])
                theoretical_sample[j,n] = np.max(np.random.chisquare(2*b, size=len(bps.ps[f_ind:])))/float(b)
        if plotlc:
            if n<= 5:
                savefig("%s_lc%i.png"%(namestr,n), format="png")
                close()

    if plotmean:
        psmean = psmean/float(nsims)
        figure(figsize=(12,9))
        plot(ps.freq[1:], psmean[1:], lw=2, color="black", linestyle='steps-mid')
        yscale("log")
        axis([ps.freq[1], ps.freq[-1], np.min(psmean), np.max(psmean[1:])])
        xlabel("Frequency [Hz]")
        ylabel("Leahy Power")
        title("mean periodogram out of %i simulations" %nsims)
        savefig("%s_mean_ps.png"%namestr, format="png")
        close()

        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        f_ind = np.array(ps.freq).searchsorted(fmin)
        pmin = np.min(psmean[f_ind:])
        pmax = np.max(psmean[f_ind:])
        hist(psmean[f_ind:], bins=50, range=[pmin, pmax], color="navy", alpha=0.7,
             label="Mean powers, averaged periodogram")

        sample = np.random.chisquare(2*nsims, size=len(psmean[f_ind:]))/float(nsims)
        hist(sample, bins=50, range=[pmin, pmax], color="darkred", alpha=0.7,
             label="Mean powers, theoretical prediction")

        ax.text(0.95, 0.5, r"KS-test: $p = %.3e$" %(scipy.stats.ks_2samp(psmean[f_ind:], sample)[1]),
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='black', fontsize=20)

        legend(prop={"size":18})

        xlabel("Leahy Power")
        ylabel("N(Powers)")
        title("Distribution of powers of %i averaged periodograms" %nsims)
        savefig("%s_mean_ps_dist.png"%namestr, format="png")
        close()

    if plothist:
        for j,b in enumerate(pbins):
            fig = figure(figsize=(12,9))
            ax = fig.add_subplot(111)
            pmin = np.min(maxpowers[j,:])
            pmax = np.max(maxpowers[j,:])
            hist(maxpowers[j,:], bins=50, range=[pmin, pmax], color="navy", alpha=0.7,
                 label="Maximum Powers, simulated light curve")

            #rv = scipy.stats.poisson(2*b)
            #theoretical_sample = rv.rvs(size=nsims)
            hist(theoretical_sample[j,:], bins=50, range=[pmin, pmax], color="darkred", alpha=0.7,
                 label="maximum powers, theoretically expected distribution")

            ax.text(0.95, 0.5, r"KS-test: $p = %.3e$" %(scipy.stats.ks_2samp(maxpowers[j,:], theoretical_sample[j,:])[1]),
                        verticalalignment='top', horizontalalignment='right',
                        transform=ax.transAxes,
                        color='black', fontsize=20)



            xlabel("Maximum Leahy-normalised power")
            ylabel("N(powers)")
            legend(prop={'size':18})
            title(r"$n_{\mathrm{photons}} = %i$, binning factor $b = %i$" %(int(np.sum(cr)),b))
            savefig("%s_b%i_maxpow_dist.png" %(namestr,b), format="png")
            close()



    return maxpowers, theoretical_sample, df



def simulate_burst(amp=10, scale=0.001, ncomp=1, bkg_cr=0.05, nsims=10000, fmin=1000.0, plotlc=False, plothist=True,
                   pbins=[1,5,10,20,50], flat=False):


    print("pbins: " + str(pbins))
    ## duration of a spike
    tspike = 0.05

    if ncomp == 1:
        tseg = 0.3
    if ncomp == 2:
        tseg = 0.3
    if ncomp == 3:
        tseg = 0.5

    dt = 0.5/2048.0

    tseg = np.max([0.1,4*scale])
    
    times = np.arange(0.0, tseg, dt)

    bkg = np.ones(len(times))*bkg_cr

    if flat is False:

        skew = 3.0
        print("skew %f" %skew)

        if ncomp == 1:
            params = [0.07, scale, amp, skew, bkg[0]]
        elif ncomp == 2:
            params = [0.07, scale, amp, skew, 0.12, scale, amp, skew, bkg[0]]
        elif ncomp == 3:
            params = [0.07, scale, amp, skew, 0.15, scale, amp, skew, 0.3, scale, amp, skew, bkg[0]]

        wordlist = [word.TwoExp for n in xrange(ncomp)]

        bd = burstmodel.BurstDict(times, bkg, wordlist)

        p = parameters.TwoExpCombined(params, ncomp, log=False, bkg=True)

        print("p.bkg: " + str(p.bkg))

        model_counts = bd.model_means(p)

        print("sum of model_counts: " + str(np.sum(model_counts)))

    else:

        model_counts = bkg

    maxpowers, theoretical_sample, df = simulate_powers(times, model_counts, nsims=nsims, pbins=pbins, fmin=fmin,
                                                    namestr="burstsim_s%i_scale%f_amp%i"%(ncomp, scale,amp),
                                                    plotmean=False, plotlc=plotlc, plothist=plothist)

    return maxpowers, theoretical_sample, df


def compute_quantile_difference(maxpowers, theo_sample, interval=0.99):

    all_intervals = [0.5-interval/2.0, 0.5+interval/2.0]

    cl_diff = []

    for m,t in zip(maxpowers, theo_sample):
        qmaxp = quantiles(m, all_intervals)
        qtheo = quantiles(t, all_intervals)

        cl_diff.append(qmaxp[1]-qtheo[1])

    return cl_diff




def simulate_many_bursts(nspikes=[1], peak_count_rates=[3,5,10,20,50,100,200], scales=[0.005,0.01,0.02,0.03],
                         pbins=[1,5,10,20,50], bkg=0.01, nsims=1000, flat=False):


    #nspikes = [1,2,3]
    #nspikes = [1]
    #peak_count_rates = [3,5,10,20,50,100,200]
    #pbins = [1,5,10,20,50]
    #scales = [0.005, 0.01, 0.02, 0.03]

    print("pbins: " +str(pbins))

    for n in nspikes:
        pvals, cl_all = [], []
        for s in scales:
            pval_temp, cl_temp = [], []
            for p in peak_count_rates:
                if p < bkg:
                    p = 0
                else:
                    p = p-bkg
                maxpowers, theo_sample, df = simulate_burst(amp=p, scale=s, ncomp=n, bkg_cr=bkg, nsims=nsims,
                                                            fmin=1000.0, pbins=pbins, flat=flat, plothist=False, plotlc=False)

                print("pbins: " + str(pbins))
                assert np.shape(maxpowers)[0] == np.shape(theo_sample)[0] == len(pbins), "Something wrong"
                cl_temp.append(compute_quantile_difference(maxpowers, theo_sample, interval=0.99))
                pval_temp.append([scipy.stats.ks_2samp(maxpowers[j,:], theo_sample[j,:])[1] for j in range(len(pbins))])
            pvals.append(pval_temp)
            cl_all.append(cl_temp)
    return np.array(pvals), np.array(cl_all), df




def plot_pvals_burstsims(pval=None, flat=False, nsims=2000):

    """
    Plot KS-test p-values for simulated bursts versus theoretical expectation (chisquared with 2M
    degrees of freedom, M = number of averaged frequency bins) for various burst amplitudes

    p-values for:
        - 1 spike
        - 4 background parameters: bkg = [0.01, 0.1, 1.0, 10.0]
        - 5 rise times: scales = [0.001, 0.005, 0.01, 0.02, 0.03]
        - 5 amplitudes: peak_count_rates = [3,5,10,100,1000]
        - 10 power spectral binning factors: pbins = [1,3,5,10,20,50,70,100,200,300]

    spawns a whole number of plots:
        - first five light curves + periodograms for each parameter combination,
        - distribution plots for the maximum power in each periodogram with f > 1000 Hz, compared the the
          maximum powers derived from a theoretical distribution of the binned periodogram


    """


    if flat is False:
        bkg = [0.01, 0.1, 1.0, 5.0, 10.0]
        #bkg = [0.01, 0.1, 1.0, 10.0]
        scales = [0.001, 0.01, 0.03]
        peak_count_rates = [5,10,50]
        pbins = [1,5,10,20,50]


    else:
        scales = [0.1]
        peak_count_rates = [0.0]
        bkg = [0.05, 0.1, 1.0, 5.0, 10.0, 50.0]
        pbins = [1,5,10,20,50]

    if pval is None:
        for b in bkg:
            pval, cl_all, df = simulate_many_bursts([1], peak_count_rates, scales, pbins, b, nsims=nsims, flat=flat)

            print("shape pval: " + str(np.shape(pval)))
            print("(should be: " + str([len(scales), len(peak_count_rates), len(pbins)]))

            print("shape cl: " + str(np.shape(cl_all)))
            print("(should be: " + str([len(scales), len(peak_count_rates), len(pbins)]))

            fig = figure(figsize=[18,8])
            subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.2, hspace=0.2)


            for pind, p in enumerate(peak_count_rates):

                ax = fig.add_subplot(1,len(peak_count_rates), pind+1)
                for i,r in enumerate(scales):
                    plot(np.array(pbins)*df, pval[i,pind,:], "-o",lw=2,
                         label=r"rise time $t_{\mathrm{rise}}= %.3f$ s"%r)
                    yscale("log")
                    legend(loc="lower right", prop={"size":14})
                    xlabel(r"Bin frequency $d\nu$ [Hz]", fontsize=16)
                    ylabel(r"2-sided KS-test p-value", fontsize=16)
                    title("Amplitude = %i" %p)
            savefig("burstsims_bkg=%f_pvals.png"%b, format="png")
            close()

            fig2 = figure(figsize=(18,8))
            subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.2, hspace=0.2)

            for pind, p in enumerate(peak_count_rates):
                ax2 = fig2.add_subplot(1,len(peak_count_rates), pind+1)
                for i,r in enumerate(scales):
                    plot(np.array(pbins)*df, cl_all[i,pind,:], "-o",lw=2,
                         label=r"rise time $t_{\mathrm{rise}}= %.3f$ s"%r)
                    #yscale("log")
                    legend(loc="lower right", prop={"size":14})
                    xlabel(r"Bin frequency $d\nu$ [Hz]", fontsize=16)
                    ylabel(r"Difference in upper 99\% confidence level", fontsize=16)
                    title("Amplitude = %i" %p)

            if flat is False:
                savefig("burstsims_bkg=%f_cl.png"%b, format="png")
            else:
                savefig("burstsims_flat_bkg=%f_cl.png"%b, format="png")

            close()

    return

