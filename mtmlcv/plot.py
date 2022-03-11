"""
functions to visualize the training results
"""
import numpy as np
import copy
import matplotlib.colors as mcolors

from collections import Counter

# from mtmlcv.separate_pe_models import JointAE

import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend("agg")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 10
colors = {}
for i, c in enumerate(mcolors.TABLEAU_COLORS.values()):
    colors[i] = np.array(mcolors.to_rgba(c), dtype=float)


class Plot:
    def __init__(self, project, data, pred, n_latent, intc, save_data=False, svg=False):

        self.project = project
        self.save_data = save_data
        self.svg = svg
        self.data = data
        self.pred = pred
        self.n_latent = n_latent
        self.intc = intc

    def plot(self, get_pe, get_label):
        """
        plot out the energy and label distribution and the latent space
        and the difference between the predicted and original potential energy for the test set and training set
        """

        self.latent_grid()
        self.predict(get_pe, get_label)

        if "pe" in self.pred:
            self.pe_parity()
            self.plot_contour(self.tilde_v, self.data["pe"], cmap='inferno', label="pe")

        if "label" in self.pred:

            self.plot_contour(
                self.tilde_n, self.data["label"], cmap=None, label="label"
            )

        if self.n_latent == 1:
            self.int_1dcontour()
        else:
            self.int_2dcontour()
            self.latent_2dcontour()

        if "label" in self.pred and self.n_latent <= 2:
            mesh, grid_shape, grid = self.latent_grid_n(20)
            tilde_n = get_label(mesh)

            if tilde_n.shape[-1] > 1:
                tilde_n = np.argmax(tilde_n, axis=-1)
            else:
                tilde_n = np.round(tilde_n)

            basins = np.unique(tilde_n)
            occ = np.zeros(mesh.shape[0])

            Xi = self.pred["latent"]
            for xi in Xi:
                ids = [np.argmin(np.abs(grid[i]-x)) for i, x in enumerate(xi)]
                values = np.array([grid[i][idx] for i, idx in enumerate(ids)])
                idx = np.where((mesh == values).all(axis=-1))[0]
                occ[idx] += 1

            matrix = []
            for basin in basins:
                for i, n in enumerate(tilde_n):
                    if n == basin and occ[i] > 0:
                        matrix += [np.hstack((n, mesh[i]))]
            matrix = np.vstack(matrix)
            np.savetxt(f"{self.project}_latent_grid.txt", matrix)


    def pe_parity(self):

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        pe_ground = self.data["pe"]
        pe_pred = self.pred["pe"]
        ids = np.where(pe_ground!=0)[0]
        ax.scatter(pe_ground[ids], pe_pred[ids], s=1)
        ax.set_xlabel("$V$")
        ax.set_ylabel("$\\tilde{V}$")
        x = np.arange(np.min(pe_pred), np.max(pe_pred), 0.01)
        ax.plot(x, x, "--")
        ax.set_title("Potential Energy Parity")

        fig.tight_layout()
        self.save_fig(f"pe_parity", fig, dict(x=pe_ground, y=pe_pred))

    def predict(self, get_pe, get_label):

        if "label" in self.pred:

            self.tilde_n = get_label(self.mesh)
            self.tilde_n = self.tilde_n.reshape(self.grid_shape+(-1,))

        if "pe" in self.pred:

            self.tilde_v = get_pe(self.mesh)
            self.tilde_v = self.tilde_v.reshape(self.grid_shape)
            print("tilde_v", self.tilde_v.shape)

    def onehot2color(self, n_array):

        # onehot
        onehot = False
        if len(n_array.shape) > 1:
            if n_array.shape[-1] > 1:
                onehot = True

        if onehot:

            nclass = n_array.shape[-1]
            new_tilde_n = np.zeros(n_array.shape[:-1]+ (4,))
            index_ones = np.ones(n_array.shape[:-1])
            for index, _ in np.ndenumerate(index_ones):
                n = n_array[index]
                # c = np.average(
                #     np.vstack([n[icomp] * colors[icomp] for icomp in range(nclass)]),
                #     axis=0,
                # )
                c = colors[int(np.argmax(n))]
                new_tilde_n[index] = c
            return new_tilde_n

        else:
            n_array = n_array.reshape([-1])

            if isinstance(n_array[0], int):
                return [colors[i] for i in n_array]
            else:
                new_n = []
                for n in n_array:
                    n1 = int(np.floor(n))
                    n2 = int(np.ceil(n))
                    dn = n - n1
                    c = colors[n2] * dn + colors[n1] * (1 - dn)
                    new_n += [c]
                return np.vstack(new_n)

    def latent_grid_n(self, ngrid):

        Xi = self.pred["latent"]

        grid = []
        for i in range(Xi.shape[1]):
            grid += [
                np.arange(
                    self.latent_min[i],
                    self.latent_max[i],
                    (self.latent_max[i] - self.latent_min[i]) / ngrid,
                )
            ]

        grid_shape = grid[0].shape
        if Xi.shape[1] > 1:

            meshgrid = np.meshgrid(*grid)
            grid_shape = meshgrid[0].shape

            flat = [meshgrid[i].reshape([-1, 1]) for i in range(Xi.shape[1])]
            mesh = np.hstack(flat)
        else:
            mesh = grid[0].reshape([-1, 1])
        return mesh, grid_shape, grid


    def latent_grid(self):

        Xi = self.pred["latent"]

        self.latent_min = np.floor(np.min(Xi, axis=0))
        self.latent_max = np.ceil(np.max(Xi, axis=0))
        self.mesh, self.grid_shape, _ = self.latent_grid_n(50)

    def plot_contour(self, tilde, real, cmap, label):

        plot_data = {}
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        latent = self.pred["latent"]

        if self.n_latent == 2:

            extent = [
                self.latent_min[0],
                self.latent_max[0],
                self.latent_min[1],
                self.latent_max[1],
            ]

            if label == "label":
                tilde = self.onehot2color(tilde)
                real = self.onehot2color(real)
            else:
                real = real.reshape([-1])
                ids = np.where(real!=0)[0]
                real = real[ids]
                latent = latent[ids]

            ax.imshow(
                tilde, extent=extent, aspect="auto", origin="low", alpha=0.5, cmap=cmap
            )
            ax.scatter(
                latent[:, 0],
                latent[:, 1],
                c=real,
                s=5,
                edgecolor="k",
                linewidths=0.1,
                cmap=cmap,
            )

            plot_data["imshow"] = tilde
            plot_data["extent"] = extent

            plot_data["X"] = latent
            plot_data["c"] = real

            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("$\\xi_2$")

        elif self.n_latent == 1:

            ax.plot(self.mesh.reshape([-1]), tilde.reshape([-1]), "k--")
            ax.scatter(latent, real.reshape([-1]), s=5, edgecolor="r", linewidths=0.1)

            plot_data["mesh"] = self.mesh
            plot_data["tilde"] = tilde
            plot_data["latent"] = latent
            plot_data["real"] = real

            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("predicted " + label)
        else:
            raise NotImplementedError()

        ax.set_title(label)
        fig.tight_layout()
        self.save_fig(label + "_contour", fig, plot_data)

    def int_1dcontour(self):

        x = self.pred["latent"].reshape([-1])

        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        for iint in range(self.intc.shape[1]):
            ax.scatter(x, self.intc[:, iint], s=1)
            ax.set_xlabel("$\\xi$")
            ax.set_ylabel(f"$c_{iint}$")

        fig.tight_layout()

        self.save_fig(f"int_1dcontour", fig, dict())

    def int_2dcontour(self):

        intc = self.intc
        x = self.pred["latent"][:, 0]
        y = self.pred["latent"][:, 1]

        ncol = int(np.ceil(intc.shape[1] / 2))
        fig, ax = plt.subplots(2, ncol)
        if ncol > 1:
            ax = ax.reshape([-1])


        for iint in range(intc.shape[1]):
            ax[iint].scatter(x, y, c=intc[:, iint], s=1, cmap="inferno")
            ax[iint].set_title(f"$v_{iint}$")
            ax[iint].set_xlabel("$\\xi_1$")
            ax[iint].set_ylabel("$\\xi_2$")

        fig.tight_layout()

        self.save_fig(f"int_2dcontour", fig, dict())

    def latent_2dcontour(self):

        for idim in range(self.n_latent):

            fig, ax = plt.subplots(figsize=(3.5, 2.5))

            plot_data = {}
            csc = ax.scatter(
                self.intc[:, 0],
                self.intc[:, 1],
                c=self.pred["latent"][:, idim],
                cmap="inferno",
            )
            plot_data[f"x"] = self.intc[:, 0]
            plot_data[f"y"] = self.intc[:, 1]
            plot_data[f"z"] = self.pred["latent"][idim]
            fig.colorbar(csc)
            ax.set_ylabel("Y")
            ax.set_xlabel("x")
            ax.set_title(f"latent_{idim}")
            fig.tight_layout()
            self.save_fig(f"latent_{idim}_contour", fig, plot_data)

    def save_fig(self, file_name, fig, plot_data=None):

        name = f"{self.project}_{file_name}"

        if self.svg:
            fig.savefig(f"{name}.svg", dpi=300, bbox_inches="tight", pad_inches=0)
        fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0)

        if plot_data is not None and self.save_data:
            np.savez(f"{name}.npz", **plot_data)
