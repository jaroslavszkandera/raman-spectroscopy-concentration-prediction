import sqlite3
from typing import Tuple, List
import numpy as np
from argparse import ArgumentParser
import time
import pandas as pd
from rich.progress import track

parser = ArgumentParser(
    prog="Test data generator",
    description="Data generator simulating spectrometer results",
)
parser.add_argument("--x_min", type=float, default=200, help="Wavelength min")
parser.add_argument("--x_max", type=float, default=700, help="Wavelength max")
parser.add_argument("--x_res", type=int, default=4096, help="Wavelength resolution")
parser.add_argument(
    "--y_noise_loc", type=float, default=700, help="Baseline noise location"
)
parser.add_argument(
    "--y_noise_sigma", type=float, default=20, help="Baseline noise distribution"
)
parser.add_argument(
    "-n",
    "--num_measurements",
    type=int,
    default=10,
    help="Number of simulated measurements",
)
parser.add_argument(
    "-o", "--output", default="spectra.csv", help="Spectra output file name"
)
parser.add_argument("-d", "--database", default="nist.sqlite", help="Input database")
args = parser.parse_args()
np.random.seed(int(time.time()))


class Component:
    """
    Spectral component with its properties

    Attributes:
        atomic_number: Atomic number of the element.
        concentration: Concentration of the element.
        peaks: Wavelengths of the peaks from peaks database.
        scale_x: Variation of the peaks locations.
        loc_y: Peaks intensities.
        scale_y: Variation of the peaks intensity.
    """

    def __init__(
        self,
        atomic_number: int,
        peaks: List[float],
        loc_y: float,
        scale_x: float = 0,
        scale_y: float = 0,
        concentration: float = 100,
    ):
        self.atomic_number = atomic_number
        self.peaks = peaks
        self.loc_y = loc_y
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.concentration = concentration

    def __repr__(self):
        if len(self.peaks) > 3:
            return f"Peak(atomic_number={self.atomic_number}, peaks={self.peaks[:3]}..., scale_x={self.scale_x}, loc_y={self.loc_y}, scale_y={self.scale_y}, concentration={self.concentration})"
        return ""


class Elem:
    def __init__(self, atomic_number: int, concentration: float):
        self.atomic_number = atomic_number
        self.concentration = concentration


def fetch_peaks(elems: Tuple[Elem, ...], db_path: str) -> List[Component]:
    """Fetches spectral peaks for given atomic numbers from the database."""
    components: List[Component] = []
    try:
        with sqlite3.connect(db_path) as con:
            for elem in elems:
                elem_peaks = con.execute(
                    "SELECT wavelength FROM lines WHERE element=?",
                    (elem.atomic_number,),
                )
                peaks: List[float] = []
                for elem_peak in elem_peaks.fetchall():
                    peaks.append(elem_peak[0])
                components.append(
                    Component(
                        atomic_number=elem.atomic_number,
                        peaks=peaks,
                        loc_y=300,
                        scale_y=0.1,
                        concentration=elem.concentration,
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    return components


def gauss(x: np.ndarray, mu: float, sigma: float, A: float = 1) -> np.ndarray:
    """
    Creates Gaussian function of x
    x - array
    mu - expected value
    sigma - square root of the variance
    A - multiplication factor
    """
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_spectrum(
    components: List[Component],
    y_noise_loc: float = args.y_noise_loc,
    x_min: float = args.x_min,
    x_max: float = args.x_max,
    x_res: int = args.x_res,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a single spectrum with dark_spectra noise and inserted peaks."""
    # Generate lambda
    xs: np.ndarray = np.linspace(x_min, x_max, x_res)

    # Generate dark_spectra noise
    ys: np.ndarray = np.random.normal(y_noise_loc, args.y_noise_sigma, len(xs))

    # Insert peaks
    for comp in components:
        for loc_x in comp.peaks:
            if x_min > loc_x or loc_x > x_max:
                print(
                    f"Not inserting peak, out of bounds ({x_min=} > {loc_x=} and {loc_x=} > {x_max=})"
                )
                continue
            x = np.clip(np.random.normal(loc=loc_x, scale=comp.scale_x), x_min, x_max)
            y = np.random.normal(loc=comp.loc_y, scale=comp.scale_y)
            sigma = np.clip(np.random.normal(loc=0.1, scale=0), 0.5, 2.5)
            peak = gauss(xs, x, sigma, y)
            ys += peak * (comp.concentration / 100)

    # Insert spikes
    """
    spikes_cnt = 3
    for _ in range(spikes_cnt):
        ys[
            np.clip(
                int(np.random.normal(loc=((x_max - x_min) / 2), scale=5)), x_min, x_max
            )
        ] *= 2
    """

    # Polynomial background
    """
    poly = (
        0.2 * np.ones(len(xs))
        + 0.005 * xs
        + 0.0005 * (xs - 500) ** 2
        + 0.000005 * (xs - 500) ** 3
    )
    ys += poly
    """

    return xs, ys


def generate_spectra(
    components: List[Component], num_spectra: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generates multiple spectra based on the same peaks."""
    xs = generate_spectrum(components)[0]
    yss = [generate_spectrum(components)[1] for _ in range(num_spectra)]
    return xs, yss


def generate_single_spectrum(
    components: List[Component],
) -> np.ndarray:
    return generate_spectrum(components)[1]


def generate_random_concentrations(n: int, total: int = 100) -> np.ndarray:
    """Generate 'n' random concentrations that sum to 'total'."""
    proportions = np.random.dirichlet(np.ones(n))
    concentrations = (proportions * total).round()
    concentrations[-1] += total - concentrations.sum()
    return concentrations


def generate_and_save_spectra(output_file: str, num_spectra: int = 100):
    """Generate multiple spectra and save to a CSV file."""
    atom_numbers = (11, 13, 14, 19)
    data = []

    for _ in track(range(num_spectra), description="Generating spectra"):
        concentrations = generate_random_concentrations(len(atom_numbers))
        components_tuple = tuple(
            [
                Elem(atom_number, concentration)
                for atom_number, concentration in zip(atom_numbers, concentrations)
            ]
        )
        components = fetch_peaks(components_tuple, "nist.sqlite")
        if not components:
            continue

        ys = generate_spectrum(components, 0, 200, 800)[1]
        row = {
            **{f"y_{i+1}": y for i, y in enumerate(ys)},
            **{
                f"atom_{atom}": conc for atom, conc in zip(atom_numbers, concentrations)
            },
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    generate_and_save_spectra("spectra.csv", num_spectra=1000)
