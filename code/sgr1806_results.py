
import numpy as np
import generaltools as gt
import classical_significances as cs
import rxte
import rxteburst
import sgr1900_results

from pylab import *

def psavg_results(datadir="./", plot_lc=True):

    """
    Get results out of averaged periodograms for SGR1806-20.
    Requires pickle output of sgr1900_results.run_average_periodograms() (in [froot]_resdict.dat)


    """
    nfolder = [5,6,8,12]


    for nf in nfolder:
        psavg = gt.getpickle(datadir+"sgr1806_psavg%i.dat"%nf)

        psavg_resdict = gt.getpickle(datadir+"sgr1806-psavg%i_resdict.dat"%nf)


        pvals = np.array([([int(i) for i in k if i.isdigit()],p) for k,p in
                          psavg_resdict["per_summary"]["bindict"].iteritems() if
                         "maxpow" in k and not "err" in k and not "sim" in k])

        pvals_ind = np.where(pvals[:,1] < 0.0005)[0]
        n_sig = len(pvals_ind)

        pvals_new = []

        for n in pvals_ind:

            p = pvals[n]
            f = ""
            for i in p[0]:
                f+=str(i)

            if float(f)*psavg.df > 150:
                continue

            freq = psavg_resdict["per_summary"]["bindict"]["bmaxfreq"+f]

            power = psavg_resdict["per_summary"]["bindict"]["bmax"+f]

            binps = psavg.rebinps(float(f)*psavg.df)
            npowers = len(binps.freq)


            if freq > 250.0:
                print("power: %f" %power)
                print("bin: %f" %float(f))
                pval_classical = cs.pavnosig(power, float(f)*psavg.m)

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

                            binps = psavg.rebinps(float(f)*psavg.df)
                            n_freq = np.array(binps.freq).searchsorted(250.0)

                            psmean = np.mean(binps.ps[n_freq:])
                            psvar = np.var(binps.ps[n_freq:])
                            npowers = len(binps.ps[n_freq:])
                            theovar = 4.0/float(f)

                            ax = fig.add_subplot(111)

                            ax.text(0.95, 0.9, r"$\mu = %.3f$, $\sigma^2 = %.3f$" %(psmean, psvar) + " (%.2f)" %theovar +
                                           "\n" + "for %i powers" %npowers,
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
                            title("Averaged periodogram %i, p = %s" %(n, p[-1]))
                            savefig("sgr1806_psavg%d_b%s.png"%(nf,f), format="png")
                            close()

    return


def run_psavg_sims(bursttimefile):

    """
    Make 1000 averaged periodograms from randomly sampled sets of bursts to check
    whether the measured power and the frequency of the "real" QPO are common for
    these bursts

    requires:
        - averaged periodograms sgr1806_psavg[n].dat, where n is a number out of
        - psno = [5,6,8,12]

        - data folders with original data to extract bursts:
        datadirs = ["P20165/20165-01-01-000", "P20165/20165-01-01-001", "P20165/20165-01-01-002",
                    "P10223/10223-01-03-01",  "P10223/10223-01-03-010" ]
    """

    nfolder = [5,6,8,12]
    datadirs = ["P20165/20165-01-01-000", "P20165/20165-01-01-001", "P20165/20165-01-01-002",
                "P10223/10223-01-03-01",  "P10223/10223-01-03-010" ]

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

    psno = [5,6,8,12]
    m_all = [30, 23, 23, 50]

    for n,m in zip(psno, m_all):
        psavg_all = sgr1900_results.make_randomly_sampled_periodograms(datadirs, bursttimefile, m, n=1000,
                                                                       save_step=100, fileroot="sgr1806_psavg%i"%n,
                                                                       data_sorted=data_sorted, t0_sorted=t0_sorted,
                                                                       pcus_sorted=pcus_sorted, tend_sorted=tend_sorted,
                                                                       tstart_sorted=tstart_sorted,
                                                                       unbary_sorted=unbary_sorted)

    return


def results_psavg_sims():
    """
    SGR1806 version of the function in sgr1900_results of the same name.
    Runs the same code, although with different filenames.
    For SGR1806,
        - psno = [5,6,8,12]
        - froot = "sgr1806"



    Posterior means can be an array with all the posterior means of the averaged periodogram, in the
    same order as the numbers in psno.

    For SGR 1806-20: posterior_means = [[1.18040327516, 7.55106444832, 3.27420103073, 3.51998795534, 0.67212630002],
                                        [0.619197296326, 6.49420626987, 2.22495505139, 2.27682390376, 0.678172183554],
                                        [0.856628471666, 5.94732402905, 3.97580346111, 3.85788708662, 0.690090617623],
                                        [0.774906025167, 7.34275742443, 2.69729821931, 2.97994334746, 0.663015258594]]
    """
    posterior_means = [[1.18040327516, 7.55106444832, 3.27420103073, 3.51998795534, 0.67212630002],
                       [0.619197296326, 6.49420626987, 2.22495505139, 2.27682390376, 0.678172183554],
                       [0.856628471666, 5.94732402905, 3.97580346111, 3.85788708662, 0.690090617623],
                       [0.774906025167, 7.34275742443, 2.69729821931, 2.97994334746, 0.663015258594]]


    sgr1900_results.results_psavg_sims(posterior_means, [5,6,8,12], "sgr1806")

    return
