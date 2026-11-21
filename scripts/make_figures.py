#!/usr/bin/env python3
"""Regenerate the paper's figures and numerical tables from analytic exponents.

Run from any directory: python scripts/make_figures.py
All source models are illustrative and share a unit mean and the same leading
bright-source tail. No Monte Carlo draws or fitted source-population data enter.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pta_distribution import density_curves, model_names, model_labels, source_shape

FIG = ROOT / "figures"
DATA = ROOT / "data"
NS = (10, 100, 1000, 10000)
MODEL_COLORS = {"paper": "#126782", "sphere": "#bc4b36", "gaussian": "#5c6d31", "mixture": "#775ba0"}
ORDER_COLORS = {None: "#152a35", 0: "#8d8c91", 1: "#21918c", 2: "#d88c18", 3: "#b14c76"}
ORDER_LABELS = {None: "Exact", 0: "Stable limit", 1: r"Through $N^{-1/3}$", 2: r"Through $N^{-2/3}$", 3: r"Through $N^{-1}$"}
ORDER_STYLES = {None: "-", 0: "--", 1: (0, (5, 2, 1, 2)), 2: (0, (4, 2)), 3: (0, (1.3, 1.6))}


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12, "axes.labelsize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#697780", "axes.labelcolor": "#213641",
        "text.color": "#213641", "xtick.color": "#465962", "ytick.color": "#465962",
        "legend.frameon": False, "legend.fontsize": 9.4,
        "grid.color": "#e5e9ec", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "savefig.dpi": 180,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figures/{name}.pdf and .png", flush=True)


def write_csv(name, header, rows):
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def title(fig, main, subtitle):
    fig.suptitle(main, fontsize=17, fontweight="semibold", x=.075, ha="left", y=.985)
    # Use a physical offset so shorter figures retain a clear title/subtitle gap.
    fig.text(.075, .985-.34/fig.get_size_inches()[1], subtitle, fontsize=10, color="#596b74")


def source_flux_figure():
    u = np.logspace(-4, 3, 2200)
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.3))
    title(fig, "Four source populations with a common stable limit",
          "Unit mean and equal bright-source tail coefficient; the alternatives are illustrative radial profiles.")
    for model in model_names:
        f = source_shape(u, model)
        axs[0].semilogx(u, f, color=MODEL_COLORS[model], label=model_labels[model], lw=2)
        axs[1].semilogx(u, f / (4*np.sqrt(np.pi*u)), color=MODEL_COLORS[model], lw=2)
    axs[0].set(xlabel=r"Dimensionless source contribution $u=Ns$", ylabel=r"Tail-shape function $F(u)$", ylim=(-.025, 1.05))
    axs[1].set(xlabel=r"Dimensionless source contribution $u=Ns$", ylabel=r"Mean contribution per $\ln u$: $u^2\rho(u)$", ylim=(0, .55))
    for ax in axs:
        ax.grid(True, which="major")
        ax.set_xlim(1e-4, 1e3)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.51, .865), ncol=4)
    fig.subplots_adjust(top=.72, bottom=.15, left=.075, right=.985, wspace=.27)
    save_figure(fig, "source_flux_models")


def build_density_figures(dt, tmax):
    x = np.linspace(-1.8, 3.2, 501)
    curves = {}
    rows = []
    paper_rows = []
    for N in NS:
        for model in model_names:
            requested = (None, 0, 1, 2, 3) if model == "paper" else (None,)
            result = density_curves(x, N, model, orders=requested, dt=dt, tmax=tmax)
            curves[N, model] = result
            rows.extend((model, N, float(xx), float(yy)) for xx, yy in zip(x, result[None]))
            if model == "paper":
                paper_rows.extend((N, float(xx), *(float(result[o][i]) for o in (None,0,1,2,3))) for i, xx in enumerate(x))
        print(f"Computed central distributions for N={N}", flush=True)
    write_csv("source_model_comparison.csv", ["model", "N", "x", "exact_continuous_density"], rows)
    write_csv("paper_comparison.csv", ["N", "x", "exact", "stable", "order_1", "order_2", "order_3"], paper_rows)

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 8.3), sharex=True)
    title(fig, "Finite-source corrections to the paper's model",
          r"Centrally rescaled density: $X=N^{1/3}(Y-1)$. Logarithms are included at each displayed order.")
    for ax, N in zip(axs.flat, NS):
        result = curves[N, "paper"]
        for order in (None, 0, 1, 2, 3):
            ax.plot(x, result[order], color=ORDER_COLORS[order], ls=ORDER_STYLES[order], lw=2.1 if order is None else 1.8, label=ORDER_LABELS[order])
        ax.axhline(0, color="#9ba7ad", lw=.7)
        ax.set_title(rf"$N_k={N:,}$", loc="left")
        ax.set_xlim(-1.8, 3.2)
        minval = min(np.min(result[o]) for o in result)
        ax.set_ylim(min(-.015, minval*1.12), max(np.max(result[o]) for o in result)*1.09)
        ax.grid(True)
    for ax in axs[1]: ax.set_xlabel(r"Rescaled fluctuation $x$")
    for ax in axs[:,0]: ax.set_ylabel(r"Density $p_X(x)$")
    fig.legend(*axs[0,0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.51,.912), ncol=5)
    fig.text(.075, .035, "Negative portions of truncated curves are retained: the expansion is not an everywhere-positive density.", fontsize=9.2, color="#596b74")
    fig.subplots_adjust(top=.83, bottom=.11, left=.075, right=.985, hspace=.30, wspace=.22)
    save_figure(fig, "paper_convergence")

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 8.1), sharex=True)
    title(fig, "Source-model dependence at fixed effective count",
          r"Exact continuous densities share the same mean and leading tail. Gray dashes show the common stable limit.")
    for ax, N in zip(axs.flat, NS):
        for model in model_names:
            ax.plot(x, curves[N,model][None], color=MODEL_COLORS[model], lw=2, label=model_labels[model])
        ax.plot(x, curves[N,"paper"][0], color=ORDER_COLORS[0], lw=1.7, ls="--", label="Stable limit")
        ax.set_title(rf"$N_k={N:,}$", loc="left")
        ax.set_xlim(-1.8, 3.2)
        ax.set_ylim(0, max(np.max(curves[N,m][None]) for m in model_names)*1.10)
        ax.grid(True)
    for ax in axs[1]: ax.set_xlabel(r"Rescaled fluctuation $x$")
    for ax in axs[:,0]: ax.set_ylabel(r"Density $p_X(x)$")
    fig.legend(*axs[0,0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.51,.912), ncol=3)
    fig.subplots_adjust(top=.81, bottom=.10, left=.075, right=.985, hspace=.30, wspace=.22)
    save_figure(fig, "source_models")


def physical_figure(dt, tmax):
    x = np.r_[np.linspace(-3, 4, 561), np.linspace(4.05, 36, 640)]
    colors = ("#c87829", "#a94e79", "#317c8d", "#25395e")
    fig, axs = plt.subplots(2,2,figsize=(10.8,8.1),sharex=True,sharey=True)
    title(fig, "How the physical distribution narrows as the count grows",
          r"Exact density of $Y=h_{c,k}^2/\overline{h_{c,k}^2}$. The vertical axis is logarithmic.")
    for ax, model in zip(axs.flat, model_names):
        for N, color in zip(NS, colors):
            pdf = density_curves(x, N, model, orders=(None,), dt=dt, tmax=tmax)[None]
            scale=N**(1/3)
            y=1+x/scale
            py=scale*pdf
            valid=(y>0)&(py>1e-7)
            py=np.where(valid,py,np.nan)
            ax.semilogy(y,py,color=color,lw=2,label=rf"$N_k={N:,}$")
        ax.axvline(1,color="#97a1a7",lw=.8,ls=":")
        ax.set_title(model_labels[model],loc="left")
        ax.set(xlim=(.05,2.45),ylim=(1e-5,40))
        ax.grid(True,which="major")
    for ax in axs[1]: ax.set_xlabel(r"Normalized strain squared $y$")
    for ax in axs[:,0]: ax.set_ylabel(r"Density $p_Y(y)$")
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.51,.914),ncol=4)
    fig.text(.075,.025,"Only resolved continuous densities are shown; zero-source atoms in the finite-count models are negligible here.",fontsize=9,color="#596b74")
    fig.subplots_adjust(top=.83,bottom=.10,left=.075,right=.985,hspace=.29,wspace=.18)
    save_figure(fig,"physical_distributions")


def error_figure(dt, tmax, quick=False):
    x = np.linspace(-1.5,2,101)
    counts=np.logspace(1,5,9 if quick else 13)
    rows=[]
    results={}
    for model in model_names:
        errors=[]
        for N in counts:
            result=density_curves(x,float(N),model,orders=(None,0,1,2,3),dt=dt,tmax=tmax)
            row=[]
            for order in range(4):
                absolute=np.abs(result[order]-result[None])
                maxerr=float(absolute.max())
                integrated=float(np.trapezoid(absolute,x))
                rows.append((model,float(N),order,maxerr,integrated,float(result[order].min())))
                row.append(maxerr)
            errors.append(row)
        results[model]=np.array(errors)
        print(f"Computed convergence curve for {model}",flush=True)
    write_csv("convergence_errors.csv",["model","N","order","sampled_max_abs_error","central_integrated_abs_error","sampled_min_approx_density"],rows)
    fig,axs=plt.subplots(2,2,figsize=(10.8,8.1),sharex=True)
    title(fig,"Convergence of the finite-source expansion",
          r"Maximum absolute density error on 101 points in $-1.5\leq x\leq2$; each model uses its own coefficients.")
    for ax,model in zip(axs.flat,model_names):
        for order in range(4):
            ax.loglog(counts,results[model][:,order],color=ORDER_COLORS[order],ls=ORDER_STYLES[order],marker="o",markersize=3,lw=1.8,label=ORDER_LABELS[order])
        ax.set_title(model_labels[model],loc="left")
        ax.grid(True,which="major")
        ax.set_xlim(10,1e5)
    for ax in axs[1]:ax.set_xlabel(r"Effective count $N_k$")
    for ax in axs[:,0]:ax.set_ylabel("Sampled maximum absolute error")
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.51,.912),ncol=4)
    fig.subplots_adjust(top=.83,bottom=.10,left=.095,right=.985,hspace=.30,wspace=.23)
    save_figure(fig,"convergence_errors")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt",type=float,default=.02,help="Fourier grid spacing (default .02)")
    parser.add_argument("--tmax",type=float,default=160,help="Fourier upper limit (default 160)")
    parser.add_argument("--quick",action="store_true",help="Use 9 instead of 13 count values for the error curves")
    args=parser.parse_args()
    FIG.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    configure_style()
    source_flux_figure()
    build_density_figures(args.dt,args.tmax)
    physical_figure(args.dt,args.tmax)
    error_figure(args.dt,args.tmax,args.quick)
    metadata={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"N_for_comparisons":list(NS),"fourier_dt":args.dt,"fourier_tmax":args.tmax,"error_grid":{"x_min":-1.5,"x_max":2,"samples":101,"N_min":10,"N_max":100000,"N_samples":9 if args.quick else 13},"models":list(model_names),"source":"Analytic source models; deterministic Fourier inversion; no fitted or observational data."}
    (DATA/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("All figures and data tables generated.",flush=True)


if __name__=="__main__":
    main()
