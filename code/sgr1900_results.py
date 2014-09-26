
import glob
import shutil
import os
import cPickle as pickle
import numpy as np
import time as tsys
import scipy.stats
import generaltools as gt
import lightcurve
import powerspectrum
import bayes
import mle
import rxte
import rxteburst
import sample_analysis
import classical_significances as cs

import word
import parameters
import burstmodel
import dnest_sample
import mle


from pylab import *
rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)





def data_times(datadir="./", data_expression="*.asc", froot="sgr1900"):

    """
     This function takes all data files and reads out the start and tstop times of the file,
     such that I only have to read in *all* the data files once, and can then use the results
     for extdaaracting bursts from the right data file.
    """

    filenames = rxteburst.search_filenames_recursively(datadir, data_expression)

    fout = open("%s_data_times.dat"%froot, "w")
    fout.write("Start time \t end time \t MET t0 reference \n")

    for f in filenames:
        fsplit = f.split("/")
        froot = fsplit[-3]
        flen = len(fsplit[-1])
        fdir = f[:-flen]
        print("filename: %s" %f)
        print("froot: " + str(froot))

        data = rxte.RXTEData(times=None, channels=None, datafile=f, npcus=None, ra=None, dec=None,
                            emid = None, emiddir=None, bary=True)

        tstart = data.photons[0].time
        print("tstart: %.15f" %tstart)
        tend = data.photons[-1].time
        print("tend: %.15f" %tend)
        t0 = data.t0
        print("t0: %.15f" %t0)
        fout.write("%.15f \t %.15f \t %.15f \n" %(tstart, tend, t0))

    fout.close()
    return

### Gather and make a dictionary with all results:
def make_alldict(datadir = "./", alldict_filename="sgr1900_alldict.dat"):

    burstfiles = gt.search_filenames_recursively(datadir, "*burstfile.dat")
    for b in burstfiles:
        try:
            shutil.copy(b, "./")
        except:
            continue
    alldict = sample_analysis.bursts_after_analysis(datadir)

    f = open(alldict_filename, "w")
    pickle.dump(alldict, f)
    f.close()

    return alldict


### Pull out significant signals automatically:

def significant_signals(alldict, fileroot="sgr1900", plot_lc=True):


    logfile = gt.TwoPrint(fileroot + "_significantsignals.dat")

    alldictnew = {}
    counter = 0
    for key, lis in alldict.iteritems():
        alldictnew[key] = []
        for j,l in enumerate(lis):
            logfile("I am on %s, burst %i" %(key, l.bst))
            try:
                pvals = np.array([([int(i) for i in k if i.isdigit()],p) for k,p in l.per_summary["bindict"].iteritems() if
                         "maxpow" in k and not "err" in k and not "sim" in k])


                pvals_ind = np.where(pvals[:,1] < 0.005)[0]
                n_sig = len(pvals_ind)
                if n_sig >=2:

                    ### check  unbinned periodogram
                    if not l.ps_corr is None:
                        ps = l.ps_corr
                    else:
                        ps = l.ps
                    n_freq = np.array(ps.freq).searchsorted(250.0)
                    psmean = np.mean(ps.ps[n_freq:])
                    psvar = np.var(ps.ps[n_freq:])
                    npowers = len(ps.ps[n_freq:])
                    logfile("Periodogram mean mu = %f, var = %f, for %i powers" %(psmean, psvar, npowers))




                    pvals_new = []
                    for n in pvals_ind:
                        p = pvals[n]
                        f = ""
                        for i in p[0]:
                            f+=str(i)
                        if float(f)*ps.df > 150.0:
                            continue
                        freq = l.per_summary["bindict"]["bmaxfreq"+f]
                        #print("type(freq)" + str(type(freq)))
                        pvals_new.append((f, freq, p[-1]))

                        logfile("Significant detection in bin %s, at f = %f, with p = %f" %(f, freq, p[-1]))

                        freq = l.per_summary["bindict"]["bmaxfreq"+f]
                        power = l.per_summary["bindict"]["bmax"+f]
                        #print("type(freq)" + str(type(freq)))
                        binps = ps.rebinps(float(f)*ps.df)
                        npowers = len(np.where(binps.freq > 250.0)[0])

                        if freq > 250.0:
                            print("power: %f" %power)
                            print("bin: %f" %float(f))
                            pval_classical = cs.pavnosig(power, float(f))

                        else:
                            pval_classical = None

                        try:
                            plist = [f, freq[0], power, npowers, p[-1], pval_classical]
                        except IndexError:
                            plist = [f, freq, power, npowers, p[-1], pval_classical]

                        pvals_new.append(plist)

                        if plot_lc:
                            fig = figure(figsize=(24,9))
                            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.2, hspace=0.2)

                            binps = ps.rebinps(float(f)*ps.df)
                            n_freq = np.array(binps.freq).searchsorted(250.0)

                            psmean = np.mean(binps.ps[n_freq:])
                            psvar = np.var(binps.ps[n_freq:])
                            npowers = len(binps.ps[n_freq:])
                            theovar = 4.0/float(f)


                            ax = fig.add_subplot(121)
                            plot(l.lc.time, l.lc.countrate/np.max(l.pcus), lw=2, color='black', linestyle='steps-mid')
                            xlabel("Time since trigger [s]")
                            ylabel("Count rate [cts/s]")

                            ax = fig.add_subplot(122)

                            ax.text(0.95, 0.9, r"$\mu = %.3f$, $\sigma^2 = %.3f$" %(psmean, psvar) + " (%.2f)" %theovar +
                                           "\n" + "for %i powers" %npowers +
                                            "\n number of photons: %i" %len(l.times),
                                    verticalalignment='top', horizontalalignment='right',
                                    transform=ax.transAxes,
                                    color='green', fontsize=20)

                            loglog(binps.freq[1:], binps.ps[1:], lw=2, color='black', linestyle='steps-mid',
                                   label=r"Periodogram with bin frequency $\nu = %s$" %binps.df)
                            minfreq = np.min(binps.freq[1:])
                            maxfreq = np.max(binps.freq[1:])
                            minp = np.min(binps.ps[1:])/2.0
                            maxp = np.max(binps.ps[1:])*2.0
                            axis([minfreq, maxfreq, minp, maxp])
                            hlines(2.0, minfreq, maxfreq, lw=3, color='red', linestyle='dashed',
                                       label="Poisson noise level")
                            if pval_classical is not None:
                                vlines(freq, minp, maxp, lw=3, color='cyan', linestyle='dashed',
                                      label="Significant detection frequency, p = %.5e" %pval_classical)
                            else:
                                vlines(freq, minp, maxp, lw=3, color='cyan', linestyle='dashed',
                                      label="Significant detection frequency, p = %.5e" %p[-1])
                            legend(loc="lower left")

                            xlabel("Frequency [Hz]", fontsize=22)
                            ylabel("Leahy Power", fontsize=22)
                            title(r"Periodogram for ObsID $%s$, $t_{\mathrm{start}} = %.2f$, p = %f "
                                    %(key, l.bst, p[-1]))
                            savefig(key + "_" + "%.2f" %l.bst + "_f%s" %f + ".png", format="png")
                            close()

                    l.sig_signals = pvals_new
                    alldictnew[key].append(l)
                    counter += 1
            except AttributeError:
                logfile("No Periodicity Search in burst %s, %f" %(key, l.bst))
                continue
    logfile("%i bursts with significant detections in total!" %counter)
    return alldictnew


def significant_signals_distributions(alldict, namestr="sgr1900", sigma=4, nbursts = None):

    """
    Play with significant signals.
    alldict is a dictionary with *all* analysed bursts

    """

    ### Make a plot with the distribution of p-values

    fig = figure(figsize=(24,12))
    ax = fig.add_subplot(121)
    pvals = []


    counter = 0
    for key, lis in alldict.iteritems():
        for l in lis:
            try:
                pvals_burst = np.array([([int(i) for i in k if i.isdigit()],p) for k,p in l.per_summary["bindict"].iteritems() if
                                "maxpow" in k and not "err" in k and not "sim" in k])

                pvals_burst = pvals_burst[:,1]
                pvals.extend(pvals_burst)
                counter += 1
            except AttributeError:
                continue


    if nbursts is None:
        nbursts = counter

    pvals = np.array(pvals).flatten()
    pval_min = np.min(pvals)
    pval_max = np.max(pvals)
    n, bins, patches = hist(pvals, bins=100, color='navy', range=[pval_min, pval_max], normed=False,
                            histtype='stepfilled')
    xlabel("Posterior p-value", fontsize=20)
    ylabel("N(p-value)")

    ax = fig.add_subplot(122)
    log_pvals = np.log10(pvals)
    ### if pvalue is 0, then log10(pvalue) == inf; deal with that:
    pvals_inf = np.where(np.isinf(log_pvals))[0]
    for i in pvals_inf:
        log_pvals[i] = -3.0
    n, bins, patches = hist(log_pvals, bins=100, color='darkred',
                            #range=[np.min(log_pvals), np.max(log_pvals)], normed=False,
                            histtype='stepfilled')
    xlabel(r"$\log_{10}{(\mathrm{p-value})}$" , fontsize=20)
    ylabel("N(p-value)")

    savefig("%s_pvalues_all.png" %namestr, format='png')
    close()

    pvals_all = []
    alldict_sig = {}
    alldict_todo = {}

    nbins = 14.0 ## number of binning factors searched
    nsigma = [3.1514872, 21.977895, 370.398, 15767.0, 1744278.0, 506797346.0, 390682215445.0]
    try:
        p_threshold = 1.0/(nsigma[sigma-1]*float(nbursts)*nbins)
        print("Significance threshold is: %.5e" %p_threshold)
    except IndexError:
        raise Exception("Please try a smaller significance threshold sigma!")


    ### First task: get out all
    for key, lis in alldict.iteritems():
        alldict_sig[key] = []
        alldict_todo[key] = []
        for j,l in enumerate(lis):
            try:
                pvals = np.array([([int(i) for i in k if i.isdigit()],p) for k,p in l.per_summary["bindict"].iteritems() if
                         "maxpow" in k and not "err" in k and not "sim" in k])


                pvals_ind = np.where(pvals[:,1] < 0.005)[0]
                n_sig = len(pvals_ind)
                if n_sig >=2:

                    ### check  unbinned periodogram
                    if not l.ps_corr is None:
                        ps = l.ps_corr
                    else:
                        ps = l.bs

                    n_freq = np.array(ps.freq).searchsorted(250.0)
                    #psmean = np.mean(ps.ps[n_freq:])
                    #psvar = np.var(ps.ps[n_freq:])
                    #npowers = len(ps.ps[n_freq:])

                    pvals_new = []
                    c = 0
                    for n in pvals_ind:
                        p = pvals[n]
                        f = ""
                        for i in p[0]:
                            f+=str(i)
                        freq = l.per_summary["bindict"]["bmaxfreq"+f]
                        power = l.per_summary["bindict"]["bmax"+f]
                        #print("type(freq)" + str(type(freq)))
                        binps = ps.rebinps(float(f)*ps.df)
                        npowers = len(np.where(binps.freq > 250.0)[0])

                        if freq > 250.0:
                            print("power: %f" %p[-1])
                            print("bin: %f" %float(f))
                            pval_classical = cs.pavnosig(power, float(f))

                        else:
                            pval_classical = None

                        try:
                            plist = [f, freq[0], power, npowers, p[-1], pval_classical]
                        except IndexError:
                            plist = [f, freq, power, npowers, p[-1], pval_classical]

                        pvals_new.append(plist)
                    l.sig_signals = pvals_new

                    for p in pvals_new:
                        if p[-1] is None:
                            if c == 0:
                                alldict_todo[key].append(l)
                                c = 1
                            else:
                                continue
                        elif p[-1] <= p_threshold/p[2]:
                            if c == 0:
                                alldict_sig[key].append(l)
                                c = 1
                            else:
                                continue
                        else:
                            continue

                    pvals_all.append(pvals_new)

                alldict[key][j] = l
            except AttributeError:
                print("No Periodicity Search in burst %s, %f" %(key, l.bst))
                continue

    f = open("%s_alldict_todo.dat"%namestr, "w")
    pickle.dump(alldict_todo, f)
    f.close()

    f = open("%s_alldict_significant.dat"%namestr, "w")
    pickle.dump(alldict_sig, f)
    f.close()

    return alldict, alldict_sig, alldict_todo, pvals_all





def compare_gbm_rxte_counts():

    ### Load files with the total number of photons per bursts, for a sample of bursts
    sum_counts_rxte = gt.getpickle("sgr1900_rxte_sum_counts.dat")
    sum_counts_gbm = gt.getpickle("sgr1550_gbm_sum_counts.dat")
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    hist(log10(sum_counts_rxte), bins=50, range=[0,log10(350000)], color="navy", histtype='stepfilled', alpha=0.8,
         label="SGR 1900+14, RXTE/PCA")
    hist(log10(sum_counts_gbm), bins=50, range=[0,log10(350000)], color="darkred", histtype='stepfilled', alpha=0.8,
         label="SGR J1550-5418, Fermi/GBM")
    xlabel(r"$\log_{10}{(N_{\mathrm{counts}})}$", fontsize=20)
    ylabel(r"$N(\mathrm{bursts})$")
    title("Total number of photons per bursts for the SGR 1900+14 RXTE and SGR 1550-5418 GBM sample", fontsize=16)
    legend()
    savefig("gbm_rxte_counts_comparison.png", format="png")
    close()

    return


def model_burst(alldict, key, bst, dt, ncomp, theta_init, fitmethod="bfgs", nsim=1000):

    blist = alldict[key]
    b = [l for l in blist if l.bst == bst]
    b = b[0]

    ### string to use for filenames to be saved
    namestr = "%s_%.3f"%(key, b.bst)

    ### light curve for fitting
    lc = lightcurve.Lightcurve(b.times, timestep=dt)

    ### reset first time stamp such that the light curve starts at zero
    lc.time = lc.time - lc.time[0]

    ### make a figure with the light curve before fit
    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    plot(lc.time, lc.countrate, lw=2, color="black", linestyle='steps-mid', label="Burst light curve")

    ### make a model for the burst light curve
    wordlist = [word.TwoExp for i in xrange(ncomp)]
    bd = burstmodel.BurstDict(lc.time, lc.counts, wordlist)

    ### define posterior probability
    lpost = burstmodel.WordPosterior(lc.time, lc.counts, bd, scale_locked=False, skew_locked=False, log=True, bkg=True)

    ### to run a minimisation routine, I need the negative log-likelihood
    lpost_new = lambda x: -lpost(x)

    ### run optimisation routine
    res = scipy.optimize.minimize(lpost_new, theta_init, method=fitmethod)

    ### parameter set with output parameters
    p = parameters.TwoExpCombined(res.x, ncomp, log=True, bkg=True)

    ### light curve with the original resolution of the data
    lc = lightcurve.Lightcurve(b.times, timestep=b.lc.res)
    lc.time = lc.time - lc.time[0]
    bd = burstmodel.BurstDict(lc.time, lc.counts, wordlist)

    ### model light curve
    model_counts = bd.model_means(p)

    ### plot model light curve and save plot
    plot(lc.time, model_counts, lw=2, color="red", label="Model light curve")
    xlabel("Time since burst start [s]")
    ylabel("Count rate [cts/s]")
    title(r"Light curve, ObsID %s, $t_\mathrm{start} = %.3f$"%(key, b.bst))
    savefig(namestr+"_lc_fit.png", format="png")
    close()

    ### extract frequency resolutions and maximum powers for significant signals
    sig_signals = [s for s in b.sig_signals if isinstance(s,list)]
    freqs = [float(s[0]) for s in sig_signals]
    maxpows_obs = [float(s[2]) for s in sig_signals]

    print("theta_fit: " + str(res.x))

    ### make nsim simulated light curves and extract the maximum power > 1000 Hz
    all_maxpow = np.zeros((nsim, len(freqs)))
    all_maxpow_pure = np.zeros((nsim, len(freqs)))
    print("max countrate %f" %(np.max(model_counts*b.lc.res)))
    for n in xrange(nsim):
        ### poissonify light curve
        mpoisson = np.array([np.random.poisson(c) for c in model_counts*b.lc.res])
        pure_poisson = np.random.poisson(np.mean(b.lc.countrate)*b.lc.res, size=len(mpoisson))
        ps = powerspectrum.PowerSpectrum(lc.time, counts=mpoisson, norm='leahy')
        ps_pure = powerspectrum.PowerSpectrum(lc.time, counts=pure_poisson, norm='leahy')
        print("n %i" %n)
        for i,f in enumerate(freqs):
            bps = ps.rebinps(ps.df*f)
            bps_pure = ps_pure.rebinps(ps_pure.df*f)
            f_ind = np.array(bps.freq).searchsorted(1000.0)
            #print(bps.ps)
            #print("i %i" %i)
            #print("maxpow: %f" %np.max(bps.ps[f_ind:]))
            all_maxpow[n,i] = np.max(bps.ps[f_ind:])
            all_maxpow_pure[n,i] = np.max(bps_pure.ps[f_ind:])

    f = open(namestr + "simulated_maximum_powers.dat", "w")
    pickle.dump(all_maxpow, f)
    f.close()

    ### compute distributions of maximum powers and p-values, then plot and save to file
    for i,m in enumerate(maxpows_obs):
        pv = len(np.where(all_maxpow[:,i] > m)[0])/float(nsim)
        pv_pure = len(np.where(all_maxpow_pure[:,i] > m)[0])/float(nsim)
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        n,bins,patches = hist(all_maxpow[:,i], color="navy", bins=100, histtype='stepfilled', alpha=0.8)
        n,bins,patches = hist(all_maxpow_pure[:,i], color='darkred', bins=100, histtype='stepfilled', alpha=0.8)
        vlines(m, 0, np.max(n), lw=2, color="black", linestyle='dashed', label=r"$P_\mathrm{obs} = %f$" %m)
        legend()
        ax.text(0.95, 0.5, r"pure Poisson noise $p = %f$,"%pv_pure +"\n"+ r"simulated light curve $p = %f$" %pv,
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='black', fontsize=20)
        xlabel("Maximum Leahy Power")
        ylabel("N(Maximum Leahy Power")
        title(r"Distribution of max Leahy Power $> 1000$ Hz from simulated light curves, for $\Delta\nu = %.2f$"
              %(freqs[i]*ps.df), fontsize=14)
        savefig(namestr + "_s%i.png"%(freqs[i]), format="png")
        close()

    return


def model_burst_lightcurve(lc, samplefile, fnyquist=2048.0, froot="test", p_threshold = 0.001, nsims_fit=1000,
                           nsims_max = 1e6):

    #### make a light curve with time resolution 0.5/fnyquist
    #lc = lightcurve.Lightcurve(times, timestep=0.5/fnyquist)

    ### make the corresponding, Leahy-normalised periodogram
    ps = powerspectrum.PowerSpectrum(lc, norm="leahy")

    ### adjust significance threshold for the number of frequencies searched
    p_threshold = p_threshold/len(ps.freq[1:])

    ### define the number of simulations as at least twice the significance threshold:
    nsims = int(2.0/p_threshold)
    if nsims > nsims_max:
        nsims = nsims_max
    print("Running %i simulations" %nsims)


    fitspec = mle.PerMaxLike(ps, obs=True, fitmethod="bfgs")
    fitparams = fitspec.mlest(mle.bpl, [1,3,2,2,0.6], obs=True)


    ### extract posterior samples
    #samplefile = "%s_%.3f_posterior_sample.txt"
    print("filename for posterior sample: %s" %samplefile)

    ### make model counts from posterior samples
    model_counts_all = dnest_sample.make_model_lightcurves(samplefile, times=lc.time, datadir="./")

    ### make a dummy list with the indices of all model light curves
    sample_indices = np.arange(len(model_counts_all))

    ### randomly pick from those light curves
    random_sample = np.random.choice(sample_indices, size=nsims, replace=True)

    psall = []

    ### iterate over all simulations (i.e. the indices I picked in the previous step)
    for i in random_sample:
        model_counts = model_counts_all[i]

        ### poissonify:
        cpoisson = np.array([np.random.poisson(c) for c in model_counts])

        pssim = powerspectrum.PowerSpectrum(lc.time, counts=cpoisson, norm="leahy")

        psall.append(pssim.ps)


    psall = np.array(psall)


    psall = np.transpose(psall)

    quantiles_all = []
    for p in psall:
        q = scipy.stats.mstats.mquantiles(p, prob=[0.01, 0.5, 0.99])
        quantiles_all.append(q)

    quantiles_all = np.array(quantiles_all)


    ### compute p-values
    pvals_all = []
    ### loop over powers frequency by frequency
    for i, p in enumerate(ps.ps[1:]):
        psim = psall[i+1,:]
        ### sort simulations
        psort = np.sort(psim)
        ### find power in simulations that corresponds to the power in the observed data at that frequency
        pind = psort.searchsorted(p)
        ### p-value corresponds to all samples above the threshold, divided by the total number of simulaitons
        pval = float(len(psort)-pind)/float(len(psort))
        pvals_all.append(pval)

    ### indices of all p-values that are smaller than the threshold
    pvals_ind = np.where(pvals_all < p_threshold)[0]

    sig_signals = []
    for pi in pvals_ind:
        ### read out frequency, power and p-value of each p-value that is below the threshold
        ### note that pvals_all starts with the *second* power, not the *first*, because the first
        ### corresponds to the zeroth frequency and should be excluded, this is why I need to use the index +1
        sig_signals.append([ps.freq[pi+1], ps.ps[pi+1], pvals_all[pi]])

    fig = figure(figsize=(24,9))
    subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)

    ymax = np.max([np.max(ps.ps[1:]), np.max(quantiles_all[1:,2].flatten())])
    ymin = np.min([np.min(ps.ps[1:]), np.min(quantiles_all[1:,0].flatten())])

    xmin = ps.freq[1]
    xmax = ps.freq[-1]

    fig.add_subplot(121)
    fill_between(ps.freq[1:], quantiles_all[1:,0], quantiles_all[1:,2], color="red", alpha=0.6)
    loglog(ps.freq[1:], ps.ps[1:], lw=2, color="black", linestyle="steps-mid", label="Periodogram of data")
    loglog(ps.freq[1:], quantiles_all[1:,1], lw=2, color="red", linestyle="steps-mid",
           label="mean of %i simulations"%nsims)

    axis([xmin, xmax, ymin, ymax])
    legend(loc="upper right", prop={"size":16})

    xlabel("Frequency [Hz]", fontsize=24)
    ylabel("Leahy Power", fontsize=24)

    fig.add_subplot(122)
    plot(ps.freq[1:], pvals_all, lw=2, color="black", linestyle="steps-mid", label="p-values for observed data")
    plt.yscale("log")
    xlim([ps.freq[1], ps.freq[-1]])
    hlines(p_threshold, ps.freq[1], ps.freq[-1], lw=3, color="red", linestyle='dashed',
           label="Trial-corrected significance threshold")

    xlabel("Frequency [Hz]", fontsize=24)
    ylabel("p-value", fontsize=24)
    legend(loc="upper right", prop={"size":16})

    savefig("%s_ps_quantiles.png"%froot, format="png")
    close()

    np.savetxt("%s_sig_signals.txt"%froot, sig_signals)

    return sig_signals

def fit_burst_lightcurve(lc, samplefile, fnyquist=2048.0, froot="test", p_threshold = 0.001, nsims=1000):

    #### make a light curve with time resolution 0.5/fnyquist
    #lc = lightcurve.Lightcurve(times, timestep=0.5/fnyquist)

    ### make the corresponding, Leahy-normalised periodogram
    ps = powerspectrum.PowerSpectrum(lc, norm="leahy")

    print("Running %i simulations" %nsims)


    fitspec = mle.PerMaxLike(ps, obs=True, fitmethod="bfgs")
    fitparams = fitspec.mlest(mle.bpl, [1,3,2,5,0.6], obs=True)
    bindict_data = fitparams["bindict"]
    freq_data = [bindict_data[i] for i in bindict_data.keys() if "maxfreq" in i]
    pratios_data = [bindict_data[i] for i in bindict_data.keys() if "bmax" in i and not "maxfreq" in i]

    ### extract posterior samples
    #samplefile = "%s_%.3f_posterior_sample.txt"
    print("filename for posterior sample: %s" %samplefile)

    ### make model counts from posterior samples
    model_counts_all = dnest_sample.make_model_lightcurves(samplefile, times=lc.time, datadir="./")

    ### make a dummy list with the indices of all model light curves
    sample_indices = np.arange(len(model_counts_all))

    ### randomly pick from those light curves
    random_sample = np.random.choice(sample_indices, size=nsims, replace=True)

    psall = []

    ### iterate over all simulations (i.e. the indices I picked in the previous step)
    for i in random_sample:
        model_counts = model_counts_all[i]

        ### poissonify:
        cpoisson = np.array([np.random.poisson(c) for c in model_counts])

        pssim = powerspectrum.PowerSpectrum(lc.time, counts=cpoisson, norm="leahy")

        psall.append(pssim)


    psall = np.array(psall)


    #fig = figure(figsize=(12,9))
    #ax = fig.add_subplot(111)

    #loglog(ps.freq[1:], ps.ps[1:], lw=3, color='black', linestyle='steps-mid')

    freqs_all, pratios_all = [], []
    maxind_all = []
    for k, p in enumerate(psall):
        print("I am on simulation %i" %k)
        fitsim = mle.PerMaxLike(p, obs=False, fitmethod="bfgs")
        simpars = fitsim.mlest(mle.bpl, [1,3,2,5,0.6], obs=False)

        bindict = simpars["bindict"]
        freqs = [bindict[i] for i in bindict.keys() if "maxfreq" in i]
        freqs_all.append(freqs)
        pratios = [bindict[i] for i in bindict.keys() if "bmax" in i and not "maxfreq" in i]
        pratios_all.append(pratios)
        #loglog(p.freq[1:], p.ps[1:], linestyle='steps-mid')


    ### shape of pratios_all is (nsims, nbins), where nbins is the number of frequency bins
    ### I'm looking for the distributions per frequency
    pratios_all = np.array(pratios_all)
    pratios_all = np.transpose(pratios_all)

    freqs_all = np.array(freqs_all)
    freqs_all = np.transpose(freqs_all)

    ### list of all frequency bins:
    allbins = [1,3,5,7,10,15,20,30,50,70,100,200,300,500]

    ### split up filename
    obsid = samplefile.split("_")[0]
    tstart = samplefile.split("_")[1]

    pvals_all = []
    for a, pdata, psims, fdata, fsims in zip(allbins[:len(pratios_data)], pratios_data, pratios_all,
                                             freq_data, freqs_all):

        psort = np.sort(psims)
        pind = psort.searchsorted(pdata)
        #print("psort: " + str(psort))
        pval = np.float(len(psort) - pind)/np.float(len(psort))
        pvals_all.append([fdata, pdata, pval])

        distfig = figure(figsize=(12,9))
        distax = distfig.add_subplot(111)
        print("psims: " + str(psims))
        pmin = np.min(psims)
        pmax = np.max(psims)
        print("pmin: %f"%pmin)
        print("pmax: %f"%pmax)
        n,bins,patches = hist(psims, range=[pmin, pmax],bins=30, color="cyan", histtype="stepfilled")
        vlines(pdata, 0, np.max(n), lw=3, linestyle='dashed', color="black")
        xlabel("Maximum Leahy Power", fontsize=24)
        ylabel("N(Maximum Leahy Power)", fontsize=24)
        title("ObsID %s, start time %s, binsize: %i, pval = %.2e" %(obsid, tstart, a, pval), fontsize=18)
        savefig("%s_%s_f%i_lcmodel_pdist.png"%(obsid, tstart, a), format="png")
        close()


        distfig = figure(figsize=(12,9))
        distax = distfig.add_subplot(111)
        n,bins,patches = hist(fsims, bins=30, color="cyan", histtype="stepfilled")
        vlines(fdata, 0, np.max(n), lw=3, linestyle='dashed', color="black")
        xlabel("Frequency of Maximum Leahy Power [Hz]", fontsize=24)
        ylabel("N(Frequency)", fontsize=24)
        title("ObsID %s, start time %s, binsize: %i" %(obsid, tstart, a), fontsize=18)
        savefig("%s_%s_f%i_lcmodel_fdist.png"%(obsid, tstart, a), format="png")
        close()

    np.savetxt("%s_%s_lcmodel_sigdetecs.txt"%(obsid, tstart), pvals_all)

    return pvals_all


def dnest_significances(datadir="./", p_threshold=0.001, lcmodel=True):

    posterior_files = glob.glob(datadir+"*posterior*.txt")
    print("posterior files 0-8 " + str(posterior_files))

    for f in posterior_files:
        fsplit = f.split("_")

        datafile = "%s_%s_data.dat"%(fsplit[0], fsplit[1])

        times, counts = burstmodel.read_gbm_lightcurves(datafile)

        lc = lightcurve.Lightcurve(times, counts=counts)

        if lcmodel is True:
            pvals_all = fit_burst_lightcurve(lc, f, froot="%s_%s"%(fsplit[0], fsplit[1]), nsims=10000)
        else:
            sig_signals = model_burst_lightcurve(lc, f, froot="%s_%s"%(fsplit[0], fsplit[1]), p_threshold=p_threshold)


    return



def plot_posterior_sample(froot):

    posterior_sample_file = glob.glob("*%s*_posterior_sample.txt"%froot)[0]
    data_file = glob.glob("*%s*_data_new.dat"%froot)[0]

    data = np.loadtxt(data_file)
    posterior_sample = atleast_2d(loadtxt(posterior_sample_file))

    figure()
    #ion()
    for i in xrange(0, posterior_sample.shape[0]):
      #hold(False)
      plot(data[:,0], data[:,1], 'bo')
      #hold(True)
      plot(data[:,0], posterior_sample[i, -data.shape[0]:], 'r-')
      ylim([0, 1.1*data[:,1].max()])
      draw()

    #ioff()
    show()

    figure()
    hist(posterior_sample[:,7], 20)
    xlabel('Number of Bursts')
    show()

    figure()
    pos = posterior_sample[:, 8:108]
    pos = pos[pos != 0.]
    hist(pos, 1000)
    xlabel('Time')
    title('Positions of Bursts')
    show()

    return


########################################################################################################################
####### AVERAGED PERIODOGRAMS ##########################################################################################
########################################################################################################################


def make_averaged_periodograms(allbursts=None, data_expression="*.asc", datadir="./", bursttimefile="bursts.dat",
                               fileroot="sgr1900"):

    filenames = glob.glob("*burstfile.dat")

    if allbursts is None:

        ### extract burst start times
        bstart = []
        allbursts = []
        for f in filenames:
            b = gt.getpickle(f)
            bstart.append(b.bst)
            allbursts.append(b)

        bstart = np.array(bstart)

    else:
        bstart = np.array([b.bst for b in allbursts])

    ### compute waiting times
    waiting_time = diff(bstart)

    ### group bursts, such that a waiting time of >2000s implies a new group
    p_all, bs_all, b_all = [], [], []
    ptemp = [filenames[0]]
    bstemp = [bstart[0]]
    btemp = [allbursts[0]]
    for i,w in enumerate(waiting_time):
        if w > 500:
            p_all.append(ptemp)
            ptemp = []
            bs_all.append(bstemp)
            bstemp = []
            b_all.append(btemp)
            btemp = []
        else:
            if not allbursts[i+1].blen > 26.0:
                ptemp.append(filenames[i+1])
                bstemp.append(bstart[i+1])
                btemp.append(allbursts[i+1])
            else:
                continue

    for i,(b, bs) in enumerate(zip(b_all, bs_all)):
        print("On cluster %i" %i)
        if len(b) > 20:
            print("i = %i" %(i+1))
            print("tstart_min: %f" %bs[0])
            print("tstart_max: %f" %bs[-1])
            blen = max([l.blen for l in b])
            print("blen: %f \n" %blen)

            try:
                os.mkdir("psavg%i"%i)
            except OSError:
                print("Directory exists already!")

            rxteburst.all_bursts(datadir=datadir, data_expression="*.asc", bursttimefile=bursttimefile, tstart_min=bs[0],
                             tstart_max=bs[-1], blen=blen, fdir="./psavg%i/"%i)

        else:
            print("No burst, continuing!")

    return


def average_periodograms(nfolder, fnyquist=2048.0, froot="sgr1900"):

    ### assumes I have folders of type psavg+number, where number is an int in list nfolder,
    ### in which there are bursts files to average together

    ### numbers of the folders in which I have periodograms to average
    ## for SGR1900 (for clusters with >=30 bursts): nfolder = [4, 5, 9, 10, 11, 12, 13, 14]
    ## for SGR 1806 (for clusters with >= 20 bursts): nfolder = [5, 6, 8, 12]

    for n in nfolder:
        files = glob.glob("./psavg%i/*burst.dat"%n)
        b = gt.getpickle(files[0])

        tseg = b.blen
        lc = lightcurve.Lightcurve(b.times, timestep=0.5/fnyquist, tseg=tseg)
        ps = powerspectrum.PowerSpectrum(lc, norm="leahy")
        psavg = np.zeros(len(ps.ps))

        for f in files:
            b = gt.getpickle(f)
            lc = lightcurve.Lightcurve(b.times, timestep=0.5/fnyquist, tseg=tseg)
            ps = powerspectrum.PowerSpectrum(lc, norm="leahy")
            psavg += ps.ps

        psavg = psavg/float(len(files))

        psnew = powerspectrum.PowerSpectrum()
        psnew.freq = ps.freq
        psnew.df = ps.df
        psnew.n = ps.n
        psnew.ps = psavg
        psnew.m = len(files)
        psnew.nphots = psavg[0]

        fout = open("./psavg%i/%s_psavg%i.dat"%(n,froot,n), "w")
        pickle.dump(psnew, fout)
        fout.close()

def run_average_periodograms(datadir="./", niter=500, nchain=500, nsim=10000, fitmethod="bfgs"):

    psavgfiles = gt.search_filenames_recursively(datadir, "*psavg*.dat")
    print("psavgfiles: " + str(psavgfiles))

    for f in psavgfiles:
        ps = gt.getpickle(f)
        m = ps.m
        fname = f.split("/")[-1]
        fsplit = fname.split("_")
        namestr = datadir
        for fs in fsplit[:-1]:
            namestr += fs + "-"
        namestr += fsplit[-1][:-4]
        print("namestr: %s"%namestr)
        btest = bayes.Bayes(ps, namestr=namestr, m=m)

        psfit, fakeper, model_summary = btest.choose_noise_model(mle.pl, [2,3,0.5], mle.bpl, [1,3,2,3,0.5], nchain=nchain, niter=niter, nsim=nsim, fitmethod=fitmethod)

        if not psfit:
            print("Analysis of burst " + str(namestr) + " failed. Returning ...")
            continue

        else:
            if model_summary["p_lrt"][0] < 0.05:
                print("Model not adequately fit by a power law! Using broken power law instead!")
                model = mle.bpl
                psfit = getattr(psfit, str(model).split()[1]+"fit")
            else:
                model = mle.pl
                psfit = getattr(psfit, str(model).split()[1]+"fit")

            per_summary = btest.find_periodicity(model, psfit["popt"], nchain=nchain, niter=niter, nsim=nsim, fitmethod=fitmethod)

            mcmc = per_summary["mcobs"]

            resdict = {"model_summary":model_summary, "per_summary":per_summary, "ps":ps, "m":m}

            fout = open("%s_resdict.dat"%namestr, "w")
            pickle.dump(resdict, fout)
            fout.close()


    return


def sgr1900_psavg_results(datadir="./"):

    """
    Requires averaged periodograms of file names sgr1900_psavg[n].dat, where n is a number
    in psavg_no.
    """

    ## number that identifies the average periodogram
    psavg_no = [4, 5, 9, 10, 11, 12, 13, 14]

    ## number of individual bursts in the averaged periodogram
    m_all = [52, 56, 70, 36, 80, 50, 45, 73]

    ## total number of averaged periodograms, for the Bonferroni correction
    n_psavg_total = 8

    ## number of bin factors searched per periodogram
    n_bins_searched = 14

    ## indvidual signals with p_bayes < 10^-4, as extracted from _findperiodicity.dat file
    freq4 = [96, 1164, 1164, 1152, 1115, 1127, 1115, 991, 1115]
    dfreq4 = [8.66, 12.38, 24.76, 37.14, 61.91, 86.67, 123, 247, 371]
    pmax4 = [2.6177, 2.55, 2.40069, 2.36, 2.2387, 2.26, 2.215, 2.1027, 2.08289]
    nbins4 = [7, 10, 20, 30, 50, 70, 100, 200, 300]
    npowers4 = [ 237, 166, 83, 56, 34, 24, 17, 9, 6]

    freq5, dfreq5, pmax5, nbins5, npowers5 = [], [], [], [], []

    freq9 = [1927, 1915, 1915, 1915]
    dfreq9= [11.26, 18, 37, 112]
    pmax9 = [2.3144, 2.2176, 2.1322, 2.0938]
    nbins9 = [30, 50, 100, 200]
    npowers9 = [182, 110, 55, 28]

    freq10, dfreq10, pmax10, nbins10, npowers10 = [], [], [], [], []

    freq11 = [55, 56, 54, 54, 52, 54, 57, 1733, 1726, 1733, 1696, 963]
    dfreq11 = [1.15, 1.976, 2.69, 3.85, 5.77, 7.7, 11.55, 19.26, 26, 38, 77, 192]
    pmax11 = [3.2564, 3.1313, 2.98810, 2.92455, 2.73449, 2.6038, 2.2974, 2.20357, 2.18588, 2.18399, 2.1323, 2.06]
    nbins11 = [3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 200, 500]
    npowers11 = [1772, 1063, 760, 532, 355, 266, 178, 107, 76, 54, 27, 11]

    freq12 = [1827, 736]
    dfreq12 = [7.366, 49]
    pmax12 = [2.4520, 2.1690]
    nbins12 = [15, 100]
    npowers12 = [278, 42]

    freq13 = [1121, 1120, 1120, 1120, 1461, 1111, 1587, 1400, 1400]
    dfreq13 = [3.26, 4.66, 7.00, 9.33, 14.00, 32, 93, 140, 233]
    pmax13 = [2.677, 2.7083, 2.55, 2.499, 2.3183, 2.21867, 2.1115, 2.09616, 2.08760]
    nbins13 = [7, 10, 15, 20, 30, 70, 200, 300, 500]
    npowers13 = [627, 439, 293, 220, 147, 63, 22, 15, 9]


    freq14 = [73, 72, 73, 677, 677, 677, 662, 736]
    dfreq14 = [2.45, 3.43, 4.9, 7.35, 9.81, 14.71, 24, 49]
    pmax14 = [2.6381, 2.5486, 2.4399, 2.3583, 2.37286, 2.219, 2.1355]
    nbins14 = [5, 7, 10, 15, 20, 30, 50, 70]
    npowers14 = [835, 597, 418, 279, 209, 140, 84, 60]

    freq_all = [freq4, freq5, freq9, freq10, freq11, freq12, freq13, freq14]
    pmax_all = [pmax4, pmax5, pmax9, pmax10, pmax11, pmax12, pmax13, pmax14]
    nbins_all = [nbins4, nbins5, nbins9, nbins10, nbins11, nbins12, nbins13, nbins14]

    for psfile, m, flist, plist, nlist in zip(psavg_no, m_all, freq_all, pmax_all, nbins_all):
        ps = gt.getpickle(datadir+"sgr1900_psavg%i.dat"%psfile)

        figure(figsize=(12,9))
        loglog(ps.freq[1:], ps.ps[1:], lw=2, color="black", linestyle="steps-mid")
        axis([ps.freq[1], ps.freq[-1], np.min(ps.ps[1:])/2.0, np.max(ps.ps[1:])*2.0])
        xlabel("Frequency [Hz]", fontsize=20)
        ylabel("Averaged Leahy Power", fontsize=20)
        title("Averaged periodogram %i, unbinned, combining %i periodograms" %(psfile, m))
        savefig(datadir+"sgr1900_psavg%i_ps.eps"%psfile, format="eps")
        close()


        for f, p, n in zip(flist, plist, nlist):
            psbin = ps.rebinps(ps.df*n)
            pval_classical = cs.pavnosig(p, float(n*m))*(n_psavg_total*n_bins_searched*len(psbin.ps))
            fig = figure(figsize=(12,9))
            ax = fig.add_subplot(111)

            loglog(psbin.freq[1:], psbin.ps[1:], lw=2, color="black", linestyle="steps-mid",
                   label=r"Periodogram with bin frequency $\nu = %.2f$" %psbin.df)
            minfreq = np.min(psbin.freq[1:])
            maxfreq = np.max(psbin.freq[1:])
            minp = np.min(psbin.ps[1:])/2.0
            maxp = np.max(psbin.ps[1:])*2.0
            axis([minfreq, maxfreq, minp, maxp])

            nind = np.array(psbin.freq).searchsorted(500.0)

            psmean = np.mean(psbin.ps[nind:])
            psvar = np.var(psbin.ps[nind:])
            npowers = len(psbin.ps[nind:])
            theovar = 4.0/float(n*m)

            ax.text(0.95, 0.5, r"$\mu = %.3f$, $\sigma^2 = %.3f$" %(psmean, psvar) + " (%.2f)" %theovar +
                                           "\n" + "for %i powers" %npowers,
                                    verticalalignment='top', horizontalalignment='right',
                                    transform=ax.transAxes,
                                    color='green', fontsize=20)

            hlines(2.0, minfreq, maxfreq, lw=3, color='red', linestyle='dashed',
                                       label="Poisson noise level")

            vlines(f, minp, maxp, lw=3, color='cyan', linestyle='dashed',
                      label="Significant detection frequency, p = %.5e" %pval_classical)

            legend(loc="upper right", prop={"size":15})

            savefig(datadir+"sgr1900_psavg%i_b%i_ps.png"%(psfile, n), format="png")
            close()

    return

def plot_psavg_components(nfolder):
    """
     Requires directories of the type psavg[n], where n is a number, such that I can
     go into the directory and plot all periodograms in it.

     for SGR1900 (for clusters with >=30 bursts): nfolder = [4, 5, 9, 10, 11, 12, 13, 14]
     for SGR 1806 (for clusters with >= 20 bursts): nfolder = [5, 6, 8, 12]

    """

    for n in nfolder:
        psall = []
        psfiles = glob.glob("psavg%i/*burst.dat"%n)

        for i,f in enumerate(psfiles):
            b = gt.getpickle(f)
            fig = figure(figsize=(12,9))
            ax = fig.add_subplot(111)
            loglog(b.ps.freq[1:], b.ps.ps[1:], lw=2, color="black", linestyle='steps-mid')
            axis([b.ps.freq[1], b.ps.freq[-1], np.min(b.ps.ps[1:])/2.0, np.max(b.ps.ps[1:])*2.0])
            xlabel("Frequency [Hz]", fontsize=20)
            ylabel("Leahy Power", fontsize=20)
            savefig("psavg%i/psavg%i_comp%i_ps.png"%(n, n, i), format="png")
            close()
            psall.append(b.ps)

        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for ps in psall:
            loglog(ps.freq[1:], ps.ps[1:], lw=2, linestyle="steps-mid")
        xlabel("Frequency [Hz]", fontsize=20)
        ylabel("Leahy Power", fontsize=20)
        title("All components in one plot!")
        savefig("psavg%i/psavg%i_all_ps.png"%(n,n), format="png")
        close()

    return



def make_randomly_sampled_periodograms(datadirs, bursttimefile, m, data_sorted= None, tstart_sorted=None, t0_sorted=None,
                                       pcus_sorted = None, tend_sorted = None, unbary_sorted=None,
                                       n=1000, save_step=20, fileroot="sgr1900"):

    """
    Make n averaged periodograms out of m randomly sampled bursts with start times and
    durations given in file bursttimefile.

    Datadirs are the directories with files which contain the bursts in bursttimefile.

    for SGR1900: datadirs = ["30410-01-02-00", "30410-01-03-00", "30410-01-04-00", "30410-01-05-00", "30410-01-06-00"]
    Assume you're in the top directory above the data directories for now.

    """

    print("generating %i simulations" %n)



    ### load burst start times
    tstart, blen_file = rxteburst.read_burst_times(bursttimefile)
    comb = [(t,b) for t,b in zip(tstart, blen_file) if b < 20.0]
    #print("comb: " + str(comb))
    #comb = zip(tstart, blen_file)
    nbursts = len(comb)


    save_counter = np.arange(n/save_step)*save_step

    print("save_counter: " + str(save_counter))


    if data_sorted is None or t0_sorted is None:

        ### load all relevant data files
        ### data_all has all the RXTEData instances
        ### tstart_all has all the start times of the observations, in time since t0
        ### tend_all has all the end times of the observations, in time since t0
        ### t0_all contains the t0 reference time in MET
        data_all, unbary_all, tstart_all, tend_all, t0_all, pcus_all, std1dir_all = [], [], [], [], [], [], []

        for d in datadirs:
            print("I am on directory %s" %d)
            files = rxteburst.search_filenames_recursively("./%s/"%d, "*1div8192*.asc")
            #print("File to use %s" %files[0])
            data = rxte.RXTEData(times=None, channels=None, datafile=files[0], npcus=None, ra=None, dec=None, emid=None, emiddir=None, bary=True)

            len_datafile = len(files[0].split("/")[-1])
            len_processed = len(files[0].split("/")[-2])
            std1dir_all.append(files[0][:-(len_datafile+len_processed+1)])

            data_all.append(np.array([p.time for p in data.photons])+data.t0)
            unbary_all.append(np.array([p.unbary for p in data.photons])+data.t0)
            tstart_all.append(data.photons[0].unbary+data.t0)
            tend_all.append(data.photons[-1].unbary+data.t0)
            t0_all.append(data.t0)
            pcus_all.append(data.pcus)

        t0_sorted, tstart_sorted, tend_sorted, data_sorted, pcus_sorted, std1dir_sorted, unbary_sorted = \
            zip(*sorted(zip(t0_all, tstart_all, tend_all, data_all, pcus_all, std1dir_all, unbary_all)))
        t0_sorted = np.array(t0_sorted)

    psavg_all = []

    tsys_start = tsys.clock()
    ### loop over all simulations
    for i in xrange(n):
        print("On simulation %i" %i)

        ### get a random sample of start times and durations
        rand_ints = np.random.choice(nbursts, size=m+30, replace=False)
        tstart_sample, blen_sample = zip(*[comb[r] for r in rand_ints])

        tstart_sample = list(tstart_sample)
        #print("len(tstart_sample): " + str(len(tstart_sample)))
        #print("blength sample: " + str(blen_sample))

        ### find burst with maximum duration
        blen = np.max(np.array(blen_sample))
        print("blength used: %f" %blen)

        ### sort start times
        tstart_sample.sort()

        b_all = []

        psavg = np.zeros(100000)
        psavg = np.array(psavg)

        nfreq = []
        j=0
        mcounter = 0
        while mcounter < m:
        #for j,t in enumerate(tstart_sample):
            t = tstart_sample[j]
            print("on periodogram %i" %j)
            j += 1

            #print("t %f" %t)
            #print("t0_sorted " + str(t0_sorted))
            #d_ind = t0_sorted.searchsorted(t)-1
            ind = [i for i,(s,e) in enumerate(zip(tstart_sorted, tend_sorted)) if s <= t <=e]
            for i,(s,e) in enumerate(zip(tstart_sorted, tend_sorted)):
                print("%f \t %f \n" %((t-s), (e-t)))
            print("ind: " + str(ind))
            if len(ind) == 0:
                continue
            else:
                d_ind = ind[0]
            #print(d_ind)
            try:

                data_temp = data_sorted[d_ind]
                unbary_temp = np.array(unbary_sorted[d_ind])
                t0_temp = t0_sorted[d_ind]

                print("t0_temp %f" %t0_temp)
                print("unbary start %f" %unbary_temp[0])
                print("t %f" %t)
                print("unbary end %f" %unbary_temp[-1])

                add_frac = 0.2
                bst = t-add_frac*blen
                bend = t+(1+add_frac)*blen

                startind = unbary_temp.searchsorted(bst)
                endind = unbary_temp.searchsorted(bend)

                if startind == endind:
                    print("SOMETHING IS NOT WORKING! GRRR!")
                    continue

                tnew = data_temp[startind:endind]

                lc = lightcurve.Lightcurve(tnew, timestep=0.5/2048.0, tseg=blen)
                ps = powerspectrum.PowerSpectrum(lc, norm="leahy")


                #b = rxte.RXTEBurst(t, blen, data_sorted[d_ind], t0_sorted[d_ind], bary=True, add_frac=0.2, fnyquist=2048.0,
                #               norm="leahy",pcus = pcus_sorted[d_ind])

                #print("b.ps " + str(b.ps))
                #b.ps_corr = b.deadtime_correction(std1dir=std1dir_sorted[d_ind])

                #b_all.append(b)

            except rxte.ZeroCountsException:
                continue


            #psavg[:len(b.ps_corr.ps)] += np.array(b.ps_corr.ps)
            #nfreq.append(len(b.ps_corr.ps))
            psavg[:len(ps.ps)] += np.array(ps.ps)
            nfreq.append(len(ps.ps))
            mcounter += 1

        print("set nfreq " + str(set(nfreq)))

        ps_avg = powerspectrum.PowerSpectrum()
        ps_avg.freq = ps.freq
        ps_avg.df = ps.df
        #ps_avg.freq = b.ps_corr.freq
        #ps_avg.df = b.ps_corr.df
        ps_avg.norm = "leahy"
        ps_avg.n = ps.n
        #ps_avg.n = b.ps_corr.n
        ps_avg.ps = psavg[:len(ps_avg.freq)]/np.float(mcounter)
        ps_avg.nphots = ps_avg.ps[0]
        ps_avg.m = mcounter

#        fout = open("psavg_sim%i.dat" %i, "w")
#        pickle.dump(psavg_dict, fout)
#        fout.close()

        psavg_all.append(ps_avg)

        print("n: %i"%n)
        if i in save_counter or i == n-1:
            fout = open("%s_psavg_sim.dat" %fileroot, "w")
            pickle.dump(psavg_all, fout)
            fout.close()


    fout = open("%s_psavg_sim.dat" %fileroot, "w")
    pickle.dump(psavg_all, fout)
    fout.close()

    tsys_end = tsys.clock()
    print("Total execution time %f"%(tsys_end-tsys_start))

    return psavg_all


def results_from_psavg_sims(posterior_means, psavg_file, psavg_sims, froot = "sgr1900_psavg_test"):

    """
     Get results out of average periodogram simulations, for the average
     periodogram in psavg_file and the corresponding simulations in psavg_sims
    """

    psavg = gt.getpickle(psavg_file)
    psavg_sim = gt.getpickle(psavg_sims)

    if posterior_means is None:
        fitspec = mle.PerMaxLike(psavg, obs=True, fitmethod="bfgs")
        fitparams = fitspec.mlest(mle.bpl, [1,6,3,3,0.6], obs=True, m=psavg.m)
        posterior_means = fitparams["popt"]

    ps_model = mle.bpl(psavg.freq, *posterior_means)

    froot = psavg_file.split(".")[0]
    psavg_no = int(froot[-1])

    fig = figure(figsize=(12,9))
    loglog(psavg.freq[1:], psavg.ps[1:], lw=2, color="black", linestyle="steps-mid",
           label=r"averaged periodogram, $m=%i$"%psavg.m)
    loglog(psavg.freq[1:], ps_model[1:], lw=2, color="red", label="posterior mean")
    axis([psavg.freq[1], psavg.freq[-1], np.min(psavg.ps)/2.0, np.max(psavg.ps[1:]*2.0)])
    xlabel("Frequency [Hz]", fontsize=18)
    ylabel("Leahy Power", fontsize=18)
    title("Average periodogram %i and posterior fit" %psavg_no)
    savefig("%s_ps.png"%froot, format="png")
    close()



    allbins = [1,3,5,7,10,15,20,30,50,70,100,200,300,500]

    bmax_all = np.zeros((len(allbins), len(psavg_sim)))
    bfreq_all = np.zeros((len(allbins), len(psavg_sim)))


    notused=0

    for i,p in enumerate(psavg_sim):
        print("On simulation %i" %i)
        fits = mle.PerMaxLike(p, obs=True, fitmethod="bfgs")
        try:
            fitsims = fits.mlest(mle.bpl, [1,6,3,3,0.5], obs=False, m=p.m)
        except TypeError:
            print("Simulation not used!")
            notused += 1
            continue
        except ValueError:
            print("simulation not used!")
            notused +=1
            continue

        print(type(p))

        #figure()
        #loglog(p.freq[1:], p.ps[1:], lw=2, color="black", linestyle="steps-mid")
        #loglog(p.freq[1:], fitsims["mfit"], lw=2, color="cyan")

        try:
            bmax_all[0,i] = fitsims["bindict"]["bmax1"]
            bfreq_all[0,i] = fitsims["bindict"]["bmaxfreq1"]

            bmax_all[1,i] = fitsims["bindict"]["bmax3"]
            bfreq_all[1,i] = fitsims["bindict"]["bmaxfreq3"]

            bmax_all[2,i] = fitsims["bindict"]["bmax5"]
            bfreq_all[2,i] = fitsims["bindict"]["bmaxfreq5"]

            bmax_all[3,i] = fitsims["bindict"]["bmax7"]
            bfreq_all[3,i] = fitsims["bindict"]["bmaxfreq7"]

            bmax_all[4,i] = fitsims["bindict"]["bmax10"]
            bfreq_all[4,i] = fitsims["bindict"]["bmaxfreq10"]

            bmax_all[5,i] = fitsims["bindict"]["bmax15"]
            bfreq_all[5,i] = fitsims["bindict"]["bmaxfreq15"]

            bmax_all[6,i] = fitsims["bindict"]["bmax20"]
            bfreq_all[6,i] = fitsims["bindict"]["bmaxfreq20"]

            bmax_all[7,i] = fitsims["bindict"]["bmax30"]
            bfreq_all[7,i] = fitsims["bindict"]["bmaxfreq30"]

            bmax_all[8,i] = fitsims["bindict"]["bmax50"]
            bfreq_all[8,i] = fitsims["bindict"]["bmaxfreq50"]

            bmax_all[9,i] = fitsims["bindict"]["bmax70"]
            bfreq_all[9,i] = fitsims["bindict"]["bmaxfreq70"]

            bmax_all[10,i] = fitsims["bindict"]["bmax100"]
            bfreq_all[10,i] = fitsims["bindict"]["bmaxfreq100"]

            bmax_all[11,i] = fitsims["bindict"]["bmax200"]
            bfreq_all[11,i] = fitsims["bindict"]["bmaxfreq200"]

            bmax_all[12,i] = fitsims["bindict"]["bmax300"]
            bfreq_all[12,i] = fitsims["bindict"]["bmaxfreq300"]

            #bmax_all[13,i] = fitsims["bindict"]["bmax500"]
            #bfreq_all[13,i] = fitsims["bindict"]["bmaxfreq500"]

        except ValueError:
            print("Simulation not used!")
            continue

    for a,f,p in zip(allbins, bfreq_all, bmax_all):
        print(len(f))
        try:
            fmax, pmax = zip(*[(f1,p1) for f1,p1 in zip(f,p) if f1>50.0])
        except ValueError:
            continue
        #print(len(fmax))

        #bfreq,bpowers = gt.rebin_lightcurve(psavg.freq, psavg.ps, n=a, type="average")
        #ps_model = mle.bpl(bfreq, *posterior_means)

        #ps_ratio = 2.0*bpowers/ps_model

        #bind = np.array(bfreq).searchsorted(50.0)

        #maxpower = np.max(ps_ratio[bind:])
        maxpower = fitparams["bindict"]["bmax%i"%a]
        maxfreq = fitparams["bindict"]["bmaxfreq%i"%a]

        #pind = np.where(ps_ratio[bind:] == maxpower)[0]

        #ftemp = bfreq[bind:]
        #maxfreq = ftemp[pind]
        print("maxfreq: %.2f" %maxfreq)

        #bfreq = fitparams["bindict"]["bmaxfreq%i"%a]
        figure()
        n,bins,patches = hist(fmax, bins=50, range=[50,2000], color="cyan", histtype="stepfilled")
        #title(r"Frequency histogram for a bin factor of %i, $\nu_{\mathrm{max}} = %.2f$"%(a, maxfreq))
        #vlines(maxfreq, 0, np.max(n), lw=3, color='black', linestyle="dashed")
        xlabel("Frequency [Hz]")
        ylabel("N(Frequency)")

        savefig("%s_b%i_freqdist.png"%(froot,a), format="png")
        close()

        #bmax = fitparams["bindict"]["bmax%i"%a]

        p_ind = np.where(pmax > maxpower)[0]
        pval = np.float(len(p_ind))/np.float(len(pmax))

        figure()
        n,bins,patches = hist(pmax, bins=50, color="cyan", histtype="stepfilled")
        vlines(maxpower, 0, np.max(n), lw=3, color="black", linestyle="dashed")
        title("Leahy Power distribution for a bin factor of %i, p-value = %f"%(a,pval))
        xlabel("Leahy Power")
        ylabel("N(Leahy Power)")
        savefig("%s_b%i_powdist.png"%(froot,a), format="png")
        close()

    print("I haven't used %i simulations! Boo!" %notused)

    return

def run_psavg_sims(bursttimefile):
    """
    Make 1000 averaged periodograms from randomly sampled sets of bursts to check
    whether the measured power and the frequency of the "real" QPO are common for
    these bursts

    requires:
        - bursttimefile: file with burst start and end times (from Ersin)
        - averaged periodograms sgr1806_psavg[n].dat, where n is a number out of
        - psno = [4, 9, 11, 12, 13, 14]

        - data folders with original data to extract bursts:
        datadirs = ["30410-01-02-00", "30410-01-03-00", "30410-01-04-00", "30410-01-05-00", "30410-01-06-00"]
    """

    datadirs = ["30410-01-02-00", "30410-01-03-00", "30410-01-04-00", "30410-01-05-00", "30410-01-06-00"]

    ### identifier of average periodogram
    psno = [4, 9, 11, 12, 13, 14]

    ### number of periodograms in each averaged spectrum
    m_all = [52, 70, 80, 50, 45, 73]

    data_all, unbary_all, tstart_all, tend_all, t0_all, pcus_all, std1dir_all = [], [], [], [], [], [], []

    for d in datadirs:
        print("I am on directory %s" %d)
        files = rxteburst.search_filenames_recursively("./%s/"%d, "*1div8192*.asc")
        if len(files) == 0:
            files = rxteburst.search_filenames_recursively("./%s/"%d, "*1div-32768s*.asc")
        if len(files) == 0:
            files = rxteburst.search_filenames_recursively("./%s/"%d, "*1div8*.asc")
        #print("File to use %s" %files[0])
        data = rxte.RXTEData(times=None, channels=None, datafile=files[0], npcus=None, ra=None, dec=None, emid=None, emiddir=None, bary=True)

        len_datafile = len(files[0].split("/")[-1])
        len_processed = len(files[0].split("/")[-2])
        std1dir_all.append(files[0][:-(len_datafile+len_processed+1)])

        data_all.append(np.array([p.time for p in data.photons])+data.t0)
        unbary_all.append(np.array([p.unbary for p in data.photons])+data.t0)
        tstart_all.append(data.photons[0].unbary+data.t0)
        tend_all.append(data.photons[-1].unbary+data.t0)
        t0_all.append(data.t0)
        pcus_all.append(data.pcus)

    t0_sorted, tstart_sorted, tend_sorted, data_sorted, pcus_sorted, std1dir_sorted, unbary_sorted = \
        zip(*sorted(zip(t0_all, tstart_all, tend_all, data_all, pcus_all, std1dir_all, unbary_all)))
    t0_sorted = np.array(t0_sorted)

    for n,m in zip(psno, m_all):
        psavg_all = make_randomly_sampled_periodograms(datadirs, bursttimefile, m, n=1000,
                                                                       save_step=100, fileroot="sgr1900_psavg%i"%n,
                                                                       data_sorted=data_sorted, t0_sorted=t0_sorted,
                                                                       pcus_sorted=pcus_sorted, tend_sorted=tend_sorted,
                                                                       tstart_sorted=tstart_sorted,
                                                                       unbary_sorted=unbary_sorted)



    return

def results_psavg_sims(posterior_means = None, psno=[4, 9, 11, 12, 13, 14], froot="sgr1900"):
    """
    This function takes the output of run_psavg_sims for all averaged periodograms
    of SGR1900 or SGR1806 and computes the corresponding plots and

    Posterior means can be an array with all the posterior means of the averaged periodogram, in the
    same order as the numbers in psno.

    For SGR1900:
    posterior_means = [[0.711146530902, 7.32701676801, 2.6777452178, 2.91304736792, 0.659628045188],
                       [1.11086516624, 7.49181011999, 3.51264421848, 4.08047087619, 0.673240944809],
                       [0.916000349594, 7.15830757391, 4.1465631204, 3.97408911765, 0.677948734909],
                       [0.958199744912, 7.14379589952, 3.34335433367, 3.75740939804, 0.67929857226],
                       [0.702449808232, 7.38958340456, 2.77430049458, 3.38653425192, 0.645308649048],
                       [0.902255328518, 7.64446791185, 3.10392608976, 3.69150858698, 0.668278462561]]

    """

    psavg_files = ["%s_psavg%i.dat"%(froot,p) for p in psno]
    psavg_sim_files = ["%s_psavg%i_psavg_sim.dat"%(froot,p) for p in psno]

    print("psavg files " + str(psavg_files))
    print("psavg simulation files " + str(psavg_sim_files))

    print("len(psavg files) %i" %len(psavg_files))
    print("len(psavg sims) %i" %len(psavg_sim_files))
    print("len(psno) %i" %len(psno))

    assert len(psavg_files) == len(psavg_sim_files) == len(psno), "Files missing! please find them!"

    if posterior_means is None:
        posterior_means = [None for i in xrange(len(psavg_files))]

    for pm, n, pf, sf in zip(posterior_means, psno, psavg_files, psavg_sim_files):
        results_from_psavg_sims(pm, pf, sf, froot = "%s_psavg%i"%(froot,n))

    return


def plot_psavg(psavg_file, posterior_means=None, bin=None, froot="sgr1900"):
    """
    Plot average periodograms and residuals. If posterior_means is None, the periodogram will be fit
    with a bent power law first.
    if bin is an integer number, a binned periodogram with bin*df will be overplotted.


    """

    psavg = gt.getpickle(psavg_file)

    if posterior_means is None:
        fitspec = mle.PerMaxLike(psavg, obs=True, fitmethod="constbfgs")
        fitparams = fitspec.mlest(mle.bpl, [1,4,2,2,0.6], obs=True, m=psavg.m)
        posterior_means = fitparams["popt"]

    ps_model = mle.bpl(psavg.freq, *posterior_means)
    ps_ratio = psavg.ps/ps_model

    froot = psavg_file.split(".")[0]
    psavg_no = int(froot[-1])

    fig = figure(figsize=(12,9))
    subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, wspace=0.1, hspace=0.0)
    s1 = subplot2grid((3,1),(0,0),rowspan=2)
    loglog(psavg.freq[1:], psavg.ps[1:], lw=2, color="black", linestyle="steps-mid",
           label=r"averaged periodogram, $m=%i$"%psavg.m)
    if not bin is None:
        bfreq, bps = gt.rebin_lightcurve(psavg.freq, psavg.ps, bin, type="average")
        loglog(bfreq[1:], bps[1:], lw=2, color="cyan", linestyle="steps-mid",
           label=r"averaged periodogram, bin frequency $d\nu = %.3f$"%(psavg.df*bin))
    loglog(psavg.freq[1:], ps_model[1:], lw=2, color="red", label="posterior mean")
    axis([psavg.freq[1], psavg.freq[-1], np.min(psavg.ps)/2.0, np.max(psavg.ps[1:]*2.0)])
    title("Average periodogram %i and posterior fit" %psavg_no)
    legend(loc="upper right", prop={"size":16})
    ylabel("Leahy Power", fontsize=18)

    s2 = plt.subplot2grid((3,1),(2,0),rowspan=1)
    plot(psavg.freq[1:], ps_ratio[1:], lw=2, color="black", linestyle="steps-mid")
    if not bin is None:
        psratio_binned = bps/mle.bpl(bfreq, *posterior_means)
        plot(bfreq[1:], psratio_binned[1:], lw=2, color="cyan", linestyle="steps-mid")
    flatline = np.ones(len(psavg.freq))
    plot(psavg.freq[1:], flatline[1:], lw=2, color="red")
    xscale("log")
    axis([psavg.freq[1], psavg.freq[-1], np.min(ps_ratio[1:])-0.1, np.max(ps_ratio[1:]+0.1)])
    xlabel(r"Frequency [Hz]", fontsize=18)
    ylabel("Data/model \n residuals", fontsize=18)
    savefig("%s_ps.png"%froot, format="png")
    close()

    return


def reduced_psavg(psavg_binds, psno = [5,6,8,12], froot="sgr1806"):
    """
    Make reduced periodograms with those bursts remove that show strong features at the frequencies
    of the potential QPO frequency, then rerun those.

    psavg_binds has the indices of strong bursts that might screw up my averaged periodogram.
    For SGR 1806-20: psavg5:  [23,24,25]
                     psavg6:  [7,18,19]
                     psavg8:  [18]
                     psavg12: [8,9,10,20,28,31,36,38,39]

    For SGR 1900+14: psavg4:  [6, 26, 44]
                     psavg9:  [9]
                     psavg11: [4, 6, 15, 18, 25, 38, 40, 46, 51, 78, 79]
                     psavg12: []
                     psavg13: []
                     psavg14: [0, 9, 11, 12, 13, 18, 27, 28, 29, 34, 52, 55, 56, 60, 68, 71]




    """

    ### indices of periodograms with the weird strong features that might screw up the average

    print("psno: " + str(psno))

    for binds,n in zip(psavg_binds, psno):

        if len(binds) == 0:
            continue

        bfiles = glob.glob("psavg%i/*burst.dat"%n)
        print("bfiles: " + str(bfiles))
        assert len(bfiles) > 0, "No files found!"

        psavg = np.zeros(100000)
        for j,f in enumerate(bfiles):
            if not j in binds:
                b = gt.getpickle(f)
                psavg[:len(b.ps.ps)] += np.array(b.ps.ps)

        psavg = psavg[:len(b.ps.freq)]
        psavg = psavg/float(len(bfiles) - len(binds))

        psavg_new = powerspectrum.PowerSpectrum()
        psavg_new.freq = b.ps.freq
        psavg_new.ps = psavg
        psavg_new.m = len(bfiles) - len(binds)
        psavg_new.df = b.ps.df
        psavg_new.norm = "leahy"
        psavg_new.n = b.ps.n
        psavg_new.nphots = psavg[0]

        f = open("%s_psavg%i_reduced.dat"%(froot,n), "w")
        pickle.dump(psavg_new, f)
        f.close()

    return



############################################################################################################
######### BROADBAND NOISE PARAMETERS #######################################################################
############################################################################################################

def make_distribution_plots(alldict):

    plind, bplind1, bplind2, breakfreq = [], [], [], []
    pl_skewness, bpl_skewness, pl_skewness_err = [], [], []
    pl_fluence, bpl_fluence = [], []
    plind_err, bpl1_err, bpl2_err = [], [], []
    pl_t90, bpl_t90 = [], []
    pl_mode, bpl1_mode, bpl2_mode = [], [], []
    pl_maxcr, bpl1_maxcr, bpl2_maxcr = [], [], []

    def sample_skewness(sample):
        smean = np.mean(sample)
        n = float(len(sample))
        m3 = (1.0/n)*np.sum((np.array(sample)-smean)**3.0)
        m2 = ((1.0/n)*np.sum((np.array(sample)-smean)**2.0))**1.5
        g1 = m3/m2
        g1_norm = np.sqrt(n*(n-1.0))/(n-2.0)*g1
        return g1_norm

    def sample_skewness_variance(sample):
        n = float(len(sample))
        top = 6.0*n*(n-1)
        bottom = (n-2.0)*(n+1.0)*(n+3.0)
        svar = top/bottom
        return svar

    def sample_mode(sample, nbins=80):

        hist, bin_edges = np.histogram(sample, bins=nbins, range = [0.0,8.0], density=True)
        maxhist = max(hist)
        maxind = np.where(hist == maxhist)[0]
        maxbin = bin_edges[maxind] + (bin_edges[1]-bin_edges[0])/2.0
        return maxbin

    plind, bplind1, bplind2, breakfreq = [], [], [], []
    pl_skewness, bpl_skewness, pl_skewness_err = [], [], []
    pl_fluence, bpl_fluence = [], []
    plind_err, bpl1_err, bpl2_err = [], [], []
    pl_t90, bpl_t90 = [], []
    pl_mode, bpl1_mode, bpl2_mode = [], [], []
    bpl_unc_t90, bpl2_unc = [], []

    c = 0
    for key, lis in alldict.iteritems():
        for l in lis:
            try:
                if l.model == mle.pl:
                    plind.append(l.per_summary['postmean'][0])
                    plind_err.append(l.per_summary['posterr'][0])
                    pl_t90.append(l.blen)
                    pl_fluence.append(len(l.times))
                    pl_skewness.append(sample_skewness(l.mcmc.mcall[0]))
                    pl_skewness_err.append(sample_skewness_variance(l.mcmc.mcall[0]))
                    pl_mode.append(sample_mode(l.mcmc.mcall[0], nbins=160))
                elif l.model == mle.bpl:
                    bplind1.append(l.per_summary['postmean'][0])
                    #bplind2.append(l.per_summary['postmean'][2])
                    if l.per_summary['postmean'][2] > 10:
                        bpl_unc_t90.append(l.blen)
                        bpl2_unc.append(l.per_summary['postquantiles'][2][0])
                        bplind2.append(l.per_summary['postquantiles'][2][0])
                    else:
                        bplind2.append(l.per_summary['postmean'][2])
                    bpl1_err.append(l.per_summary['posterr'][0])
                    bpl2_err.append(l.per_summary['posterr'][2])
                    bpl_fluence.append(len(l.times))
                    bpl_t90.append(l.blen)
                    breakfreq.append(l.per_summary['postmean'][3])
            except AttributeError:
                print("No model summary. Continuing ...")
                c = c+1
                continue

    print("%i bursts not used." %c)
    breakfreq = np.log10(np.exp(np.array(breakfreq)))

    #FIGURE 4: DISTRIBUTIONS OF BPL AND PL BURSTS
    print('len(pl_fluence): %i' %len(pl_fluence))
    print("len(bpl_fluence): %i" %len(bpl_fluence))
    print("min(pl_fluence) %i" %np.min(pl_fluence))
    print("max(pl_fluence) %i" %np.max(pl_fluence))

    plt.figure(figsize=(22,8.0))
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9, wspace=0.15, hspace=0.15)
    plt.subplot(1,2,1)
    hist(pl_t90, bins=20, range=[0.0,2.5], color='navy', alpha=0.7, histtype='stepfilled', label='Simple Power Law')
    hist(bpl_t90, bins=20, range=[0.0,2.5], color='darkred', alpha=0.7, histtype='stepfilled', label='Broken Power Law')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)

    plt.xlabel('Burst T90 [s]', fontsize=20)
    plt.ylabel('N(bursts)', fontsize=20)
    plt.legend(prop={'size':20}, loc='upper right')
    plt.subplot(1,2,2)
    minrange = np.min([np.min(np.log10(pl_fluence)), np.min(np.log10(bpl_fluence))])
    maxrange = np.max([np.max(np.log10(pl_fluence)), np.max(np.log10(bpl_fluence))])
    hist(np.log10(pl_fluence), bins=20, range=[minrange, maxrange], color='navy', alpha=0.7, histtype='stepfilled', label='Simple Power Law')
    hist(np.log10(bpl_fluence), bins=20, range=[minrange, maxrange], color='darkred', alpha=0.7, histtype='stepfilled', label='Broken Power Law')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)

    plt.xlabel(r'log(No of burst photons)', fontsize=20)
    plt.ylabel('N(bursts)', fontsize=20)
    plt.legend(prop={'size':20}, loc='upper right')
    plt.savefig('f4.eps', format='eps')
    plt.close()


    #### plot distributions of parameters

    fig = plt.figure(figsize=(9,21))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
    plt.subplot(4,1,1)
    hist(plind, bins=20, range=[0, 5], color='navy', histtype='stepfilled')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)

    plt.xlabel(r'Power law index $\gamma$', fontsize=18)
    plt.ylabel('N(bursts)', fontsize=20)

    plt.subplot(4,1,2)
    hist(bplind1, bins=20, range=[0, 5], color='navy', histtype='stepfilled')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)

    plt.xlabel(r'Low-frequency power law index $\gamma_1$', fontsize=18)
    plt.ylabel('N(bursts)', fontsize=20)

    plt.subplot(4,1,3)
    hist(bplind2, bins=20, range=[0, 10], color='navy', histtype='stepfilled', label='whole sample')
    hist(bpl2_unc, bins=20, range=[0,10], color='orange', histtype='stepfilled', alpha=0.7, label='lower limits only')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)

    plt.legend()
    plt.xlabel(r'High-frequency power law index $\gamma_2$', fontsize=18)
    plt.ylabel('N(bursts)', fontsize=20)
    #a = axes([.5, .35, .35, .1], axisbg='w')
    #hist(bplind2, bins=20, range=[0.0,10.0], color='navy', histtype='stepfilled')
    #ax = plt.gca()
    #for label in ax.get_xticklabels() + ax.get_yticklabels():
    #    label.set_fontsize(16)

    plt.subplot(4,1,4)
    hist(breakfreq, bins=20, range=[0.5,3.0], color='navy', histtype='stepfilled')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)

    plt.xlabel('log(Break Frequency) [Hz]', fontsize=18)
    plt.ylabel('N(Bursts)', fontsize=20)

    plt.savefig('f5.eps', format='eps')
    plt.close()


    ##### FIGURE 6: CORRELATION BETWEEN T90 and PLIND

    borders = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]

    figure(figsize=(12, 9))
    plt.errorbar(pl_t90, plind, yerr=plind_err, fmt='o', color='navy', label='Unbinned data')
    plt.axis([0.025, 2.0, 0.5, 7.0])
    pl_sample_all, pl_mean, pl_var = [], [], []

    for i in range(len(borders)-1):
        pl_temp = [p for t,p in zip(pl_t90, plind) if borders[i] <= t <= borders[i+1]]
        plmean = np.mean(pl_temp)
        plvar = np.std(pl_temp)
        pl_sample_all.append(pl_temp)
        pl_mean.append(plmean)
        pl_var.append(plvar)

    plt.errorbar(np.array(borders[:-1])+0.0125, pl_mean, yerr=pl_var, lw=2, color='orange', marker='o', label='Binned data points')
    import scipy.optimize

    def pl(x, a,b):
       return b*x**a

    popt, pcov = scipy.optimize.curve_fit(pl, borders[:-1], pl_mean, sigma=pl_var)
    fit = pl(borders[:-1], *popt)
    plot(np.array(borders[:-1])+0.0125, fit, lw=4, color='cyan', linestyle='dashed', label='Best-fit power law')
    plt.legend(prop={'size':18}, loc='upper right')
    x = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(22)

    plt.xlabel("Burst T90 [s]", fontsize=20)
    plt.ylabel('Power law index', fontsize=20)
    plt.xscale('log')
    plt.savefig("f6.eps", format="eps")
    plt.close()

    return
